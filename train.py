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
import shutil
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import UNet

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset, JAICDataModule
from torch.utils.data import DataLoader, random_split

class train_unet:
    
    def __init__(self, data_dir):
        """Initializes the UNET models and the CUDA  device

        Args:
            mode (str, optional): mode which can be normal,
                                  augmentation,temporal, and temporal_augmentation
        """
        self.data_dir = data_dir
        
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device {self.device}')
        
        # Change here to adapt to your data
        # n_channels=3 for RGB images
        # n_classes is the number of probabilities you want to get per pixel
        #   - For 1 class and background, use n_classes=1
        #   - For 2 classes, use n_classes=1
        #   - For N > 2 classes, use n_classes=N
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
                  img_scale=1,
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

        # Randomly determines the training and validation dataset
        dataset = JAICDataModule(
            batch_size=batch_size, 
            augment=augment, 
            datadir=self.data_dir, 
            scale=img_scale, 
            val_size=val_percent)
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
    
def get_args():
    """ Define the arguments that user can put in as flags in the terminal

    Returns:
        list: The list of inputs attached to args paramaters  
    """
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=1,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch_size', metavar='B', type=int, default=1,
                        help='Batch size', dest='batch')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1.0,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=0.1,
                        help='Percent of the data that is used as validation (0-1)')
    parser.add_argument('-o', '--output_dir', dest='out', type=str, default='checkpoints_test',
                        help='specify where to save the MODEL.PTH')
    parser.add_argument('-a', '--augment', dest='aug', type=bool, default=False,
                        help='specify the training augmentation')
    parser.add_argument('-d', '--data_dir', dest='dir', type=str, default='./data',
                        help='specify the directory data')
    

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    output_path = f'{args.out}/checkoints_LR_{args.lr}_BS_{args.batch}_SCALE_{args.scale}_E_{args.epochs}/'
    model = train_unet(args.dir)
    if os.path.isdir(output_path):
        shutil.rmtree(output_path) 
    os.makedirs(output_path)
    model.train_net(
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        img_scale=args.scale,
        augment=args.aug,
        val_percent=args.val, 
        dir_checkpoint=output_path)
    
    
