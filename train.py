import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import UNet

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split

class train_unet:
    
    def __init__(self, mode='normal'):
        self.dir_img = 'data/imgs/'
        self.dir_mask = 'data/masks/'
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device {self.device}')

        # Change here to adapt to your data
        # n_channels=3 for RGB images
        # n_classes is the number of probabilities you want to get per pixel
        #   - For 1 class and background, use n_classes=1
        #   - For 2 classes, use n_classes=1
        #   - For N > 2 classes, use n_classes=N
        if mode == 'temporal' or mode == 'temporal_augmentation':
            self.net = UNet(n_channels=12, n_classes=1, bilinear=True)
        else:
            self.net = UNet(n_channels=3, n_classes=1, bilinear=True)
            
        logging.info(f'Network:\n'
                     f'\t{self.net.n_channels} input channels\n'
                     f'\t{self.net.n_classes} output channels (classes)\n'
                     f'\t{"Bilinear" if self.net.bilinear else "Transposed conv"} upscaling')

        self.net.to(device=self.device)
        # faster convolutions, but more memory
        # cudnn.benchmark = True

    
    def train_net(self,
                  epochs=5,
                  batch_size=1,
                  lr=0.001,
                  val_percent=0.1,
                  save_cp=True,
                  img_scale=0.5,
                  dir_checkpoint='checkpoints/'):
        
        device = self.device
        net = self.net
        mode = self.mode
        
        file_list = [splitext(file)[0] for file in os.listdir(self.dir_img)
                    if not file.startswith('.')]
        random.shuffle(file_list)
        n_val = int(len(file_list) * val_percent)
        n_train = len(file_list) - n_val
        train_list = file_list[:n_train]
        val_list = file_list[n_train:]
        dataset_train = BasicDataset(train_list, self.dir_img, self.dir_mask, epochs, img_scale, 'train', mode)
        dataset_val = BasicDataset(val_list, self.dir_img, self.dir_mask, epochs, img_scale, 'val', mode)
        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

        writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
        global_step = 0

        logging.info(f'''Starting training:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {lr}
            Training size:   {n_train}
            Validation size: {n_val}
            Checkpoints:     {save_cp}
            Device:          {self.device.type}
            Images scaling:  {img_scale}
        ''')

        optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
        if net.n_classes > 1:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.BCEWithLogitsLoss()

        for epoch in range(epochs):
            net.train()

            epoch_loss = 0
            with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
                for batch in train_loader:
                    imgs = batch['image']
                    true_masks = batch['mask']
                    assert imgs.shape[1] == net.n_channels, \
                        f'Network has been defined with {net.n_channels} input channels, ' \
                        f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                        'the images are loaded correctly.'

                    imgs = imgs.to(device=device, dtype=torch.float32)
                    mask_type = torch.float32 if net.n_classes == 1 else torch.long
                    true_masks = true_masks.to(device=device, dtype=mask_type)

                    masks_pred = net(imgs)
                    loss = criterion(masks_pred, true_masks)
                    epoch_loss += loss.item()
                    writer.add_scalar('Loss/train', loss.item(), global_step)

                    pbar.set_postfix(**{'loss (batch)': loss.item()})

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_value_(net.parameters(), 0.1)
                    optimizer.step()

                    pbar.update(imgs.shape[0])
                    global_step += 1
                    if global_step % (n_train // (10 * batch_size)) == 0:
                        for tag, value in net.named_parameters():
                            tag = tag.replace('.', '/')
                            writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                            writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                        val_score = eval_net(net, val_loader, device)
                        scheduler.step(val_score)
                        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                        if net.n_classes > 1:
                            logging.info('Validation cross entropy: {}'.format(val_score))
                            writer.add_scalar('Loss/test', val_score, global_step)
                        else:
                            logging.info('Validation Dice Coeff: {}'.format(val_score))
                            writer.add_scalar('Dice/test', val_score, global_step)
                        if mode != 'temporal' or mode != 'temporal_augmentation':
                            writer.add_images('images', imgs, global_step)
                        if net.n_classes == 1:
                            writer.add_images('masks/true', true_masks, global_step)
                            writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)

            if save_cp:
                try:
                    os.mkdir(dir_checkpoint)
                    logging.info('Created checkpoint directory')
                except OSError:
                    pass
                torch.save(net.state_dict(),
                           dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
                logging.info(f'Checkpoint {epoch + 1} saved !')

        writer.close()
        return val_score