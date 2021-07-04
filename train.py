####################################################################
# Main class for running the training based on each give paramater #
####################################################################

import argparse
import logging
import os
import sys
import random

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import UNet

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset, JAICDataModule
from torch.utils.data import DataLoader, random_split

class train_unet:
    
    def __init__(self):
        """Initializes the UNET models and the CUDA  device

        Args:
            mode (str, optional): mode which can be normal,
                                  augmentation,temporal, and temporal_augmentation
        """
<<<<<<< HEAD
        self.dir_img = '../process_midrc_small/dicoms'
        self.dir_mask = '../process_midrc_small/labels'
        self.data_dir = '../process_midrc_small'
=======
        self.dir_img = 'data/dicoms/'
        self.dir_mask = 'data/labels/'
>>>>>>> cf1d56122735f32408ef93d41cbdc996f24ef3de
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device {self.device}')
        
        # Change here to adapt to your data
        # n_channels=3 for RGB images
        # n_classes is the number of probabilities you want to get per pixel
        #   - For 1 class and background, use n_classes=1
        #   - For 2 classes, use n_classes=1
        #   - For N > 2 classes, use n_classes=N
<<<<<<< HEAD

        self.net = UNet(n_channels=1, n_classes=1, bilinear=True)
=======
        if mode == 'temporal' or mode == 'temporal_augmentation':
            self.net = UNet(n_channels=12, n_classes=1, bilinear=True)
        else:
            self.net = UNet(n_channels=1, n_classes=1, bilinear=True)
>>>>>>> cf1d56122735f32408ef93d41cbdc996f24ef3de
            
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
                  augment=False,
                  dir_checkpoint='checkpoints/'):
        """Runs training based on paramaters on the data

        Args:
            epochs (int, optional): Number of epochs to run the model. Defaults to 5.
            batch_size (int, optional): Batchsize number to be taken from model. Defaults to 1.
            lr (float, optional): Learning rate for stepping. Defaults to 0.001.
            val_percent (float, optional): Percentage of data to be taken for validation. Defaults to 0.1.
            save_cp (bool, optional): Save the weights or not. Defaults to True.
            img_scale (float, optional): Scale percentage of the original image to use. Defaults to 0.5.
            dir_checkpoint (str, optional): path to save the trained weights. Defaults to 'checkpoints/'.

        Returns:
            int: best validation score recorded in one training
        """
        
        device = self.device
        net = self.net
<<<<<<< HEAD
        
=======
        mode = self.mode

>>>>>>> cf1d56122735f32408ef93d41cbdc996f24ef3de
        # Randomly determines the training and validation dataset
#         file_list = [os.path.splitext(file)[0] for file in os.listdir(self.dir_img)
#                     if not file.startswith('.')]
#         random.shuffle(file_list)
#         n_val = int(len(file_list) * val_percent)
#         n_train = len(file_list) - n_val
#         train_list = file_list[:n_train]
#         val_list = file_list[n_train:]

#         dataset_train = BasicDataset(train_list, self.dir_img, self.dir_mask, epochs, img_scale, 'train', mode)
#         dataset_val = BasicDataset(val_list, self.dir_img, self.dir_mask, epochs, img_scale, 'val', mode)

        dataset = JAICDataModule(batch_size=batch_size, augment=augment, datadir=self.data_dir)
        train_loader = dataset.train_dataloader()
        val_loader = dataset.val_dataloader()
        n_train = len(train_loader)
        n_val = len(val_loader)
        
        # Tensorboard initialization
        writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
        global_step = 0
        val_score_list = []
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

        # Gradient descent method
        optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
        if net.n_classes > 1:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.BCEWithLogitsLoss()

        for epoch in range(epochs):
            net.train()
            epoch_loss = 0
#             import pdb;pdb.set_trace()
            # Progress bar shown on the terminal
            with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
                for batch in train_loader:
                    imgs = batch['data']
                    true_masks = batch['label']
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
                    
                    # Validation phase
                    if global_step % (n_train // (10 * batch_size)) == 0:
                        for tag, value in net.named_parameters():
                            tag = tag.replace('.', '/')
                            writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                            writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                        val_score = eval_net(net, val_loader, device)
                        val_score_list.append(val_score)
                        scheduler.step(val_score)
                        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                        if net.n_classes > 1:
                            logging.info('Validation cross entropy: {}'.format(val_score))
                            writer.add_scalar('Loss/test', val_score, global_step)
                        else:
                            logging.info('Validation Dice Coeff: {}'.format(val_score))
                            writer.add_scalar('Dice/test', val_score, global_step)

                        writer.add_images('images', imgs, global_step)
                        if net.n_classes == 1:
                            writer.add_images('masks/true', true_masks, global_step)
                            writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)

            if save_cp: #saves the trained weights
                try:
                    os.mkdir(dir_checkpoint)
                    logging.info('Created checkpoint directory')
                except OSError:
                    pass
                torch.save(net.state_dict(),
                           dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
                logging.info(f'Checkpoint {epoch + 1} saved !')

        writer.close()
        return max(val_score_list)
