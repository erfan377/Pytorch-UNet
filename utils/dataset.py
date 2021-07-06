#########################################
# Dataset loader class from the folders #
#########################################

from os.path import splitext
from os import listdir
import numpy as np
import pandas as pd
from glob import glob
import torch
import os
# from torch.utils.data import Dataset
# from torch.utils.data import DataLoader
import logging
from PIL import Image
import albumentations as A
import torchvision.transforms.functional as TF
import pdb

import rising.transforms as rtr
from rising.loading import Dataset, DataLoader
from rising.random import UniformParameter
from utils.constants import *

class MIDRCDataset(Dataset):
  """A dataset consisting of COVID and non-COVID chest x-rays based on CheXpert and RICORD"""

  def __init__(self, df, split='train', scale=1):
    """Create a dataset of COVID+/- chest CTs and split into train, val and test.
      Keyword arguments:
      df -- df including the data and metadata
      split -- either a train, val, or test split of the data
    """

    self.df = df.get(split)
    self.scale = scale


  def __len__(self):
    """Return the number of samples in the dataset."""

    return len(self.df)

  def __getitem__(self, idx):
    """Generate a tuple of a lung chest x-ray and its associated labels.
      Keyword arguments:
      idx -- index of the chest x-ray in the dataset split
    """

    # image
    img_path, label_path = self.df['path'].iloc[idx], self.df['label_path'].iloc[idx]
    img = torch.load(img_path)
    label = torch.zeros_like(img) # In case we have no labels
    
    img = img.type(torch.float32).expand(3, img.shape[1], img.shape[2])

    if pd.isna(label_path) == False: # Load label if there's a path for it
      label = torch.load(label_path)
    
#     label = label.type(torch.float32).expand(1, label.shape[1], label.shape[2])
    
    img, label = self.preprocess(img, label)
    
    # covid/non-covid
    covid_status = 1 if self.df['label_count'].iloc[idx] > 1 else 0

    return {'data': img.float(),
           'label': label.float()}

  def preprocess(self, img, label):
        c, w, h = img.shape
        # resizes the image based on scaling factor
        newW, newH = int(self.scale * w), int(self.scale * h) 
        assert newW > 0 and newH > 0, 'Scale is too small'
        img = TF.resize(img, (newW, newH))
        label = TF.resize(label, (newW, newH))
        
        # if the scaling factor is too small the mask might lose the binary 
        # nature of it's values. We make sure in occasions of extreme scaling
        # the mask still would stay binary by rounding the resized mask values to 1 or 0
        label[label > 0] = 1

        return img, label

class JAICDataModule(DataLoader):
  """A reusable Pytorch Lightning Data Module"""

  def __init__(self, batch_size=32, augment=False, datadir='./data', scale=1.0, val_size=0.1):
    """Initialize the Data Module.
      Keyword arguments:
      batch_size -- number of samples within a mini-batch
      augment -- flag whether to apply augmentation to the training samples
    """
    assert 0 < scale <= 1, 'Scale must be between 0 and 1'
    
    self.batch_size = batch_size
    self.augment = augment
    self.num_workers = 5
    # Augmentation doesn't do anything if augment flag is false
    self.sample_transform = rtr.Compose([
        rtr.DoNothing()
    ])
    if self.augment:
      self.sample_transform = rtr.Compose([
        rtr.NormZeroMeanUnitStd(keys=('data', ))
      ])
    self.scale = scale
    self.data_dir = datadir
    self.setup(val_size)
    

  def setup(self, val_size):
    """Generate the training, validation and test splits."""
    df = pd.read_csv(os.path.join(self.data_dir, 'metadata.csv'))

    # Creating a balanced dataset of postive and negative examples
    neg_df, pos_df = df[df['label_count'] == 0], df[df['label_count'] > 0]
    minority_data = min(len(neg_df), len(pos_df))
    pos_df = pos_df.sample(minority_data)
    neg_df = neg_df.sample(minority_data)
    
    # remove incorrect cxrs (= not cxr)
    pos_df = pos_df[~pos_df['sop_instance_uid'].isin(WRONG_CRX)]
    
    train_size = 1 - val_size - 0.05
    # splits
    neg_train, neg_val, neg_test = np.split(
      neg_df.sample(frac=1),
      [int(train_size * len(neg_df)), int((train_size+val_size) * len(neg_df))],
    )

    pos_train, pos_val, pos_test = np.split(
      pos_df.sample(frac=1),
      [int(train_size * len(pos_df)), int((train_size+val_size) * len(pos_df))],
    )

    # combine train, val, test sets and reshuffle
    train = neg_train.append(pos_train).sample(frac=1).reset_index(drop=True)
    val = neg_val.append(pos_val).sample(frac=1).reset_index(drop=True)
    test = neg_test.append(pos_test).sample(frac=1).reset_index(drop=True)
    self.df_all = dict(train=train, val=val, test=test)

    self.train = MIDRCDataset(self.df_all, split='train', scale=self.scale)
    self.val = MIDRCDataset(self.df_all, split='val', scale=self.scale)
    self.test = MIDRCDataset(self.df_all, split='test', scale=self.scale)

  def train_dataloader(self):
    """Create a train dataloader."""

    transforms = rtr.Compose([
        rtr.DoNothing()
    ])
    
    if self.augment:
      transforms = rtr.DropoutCompose([
        rtr.Mirror(1, keys=('data', 'label')), # Horizontal Flip
        rtr.Mirror(2, keys=('data', 'label')), # Vertical Flip
        rtr.Rotate([0,0,UniformParameter(0, 360)], degree=True, keys=('data', 'label')),
        rtr.GammaCorrection(0.5, keys=('data', )),
        rtr.Scale(UniformParameter(0.7, 1.5), keys=('data', 'label'))
        ], dropout=0.5, shuffle=True)

    # construct loader
    return DataLoader(self.train,
                            batch_size=self.batch_size,
                            gpu_transforms=transforms,
                            shuffle=True,
                            sample_transforms=self.sample_transform,
                            num_workers=self.num_workers, 
                            pin_memory=True)

  def val_dataloader(self):
    """Create a val dataloader."""

    return DataLoader(self.val,
                            num_workers=self.num_workers,
                            batch_size=self.batch_size,
                            sample_transforms=self.sample_transform,
                            pin_memory=True)

  def test_dataloader(self):
    """Create a test dataloader."""

    return DataLoader(self.test,
                            num_workers=self.num_workers,
                            batch_size=self.batch_size,
                            sample_transforms=self.sample_transform,
                            pin_memory=True)


class BasicDataset(Dataset):
    def __init__(self, imgs_list, imgs_dir, masks_dir, epochs, scale=1, tag='train', mode='normal'):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.mode = mode
        self.ids = imgs_list
        self.tag = tag
        
        logging.info(f'Creating dataset with {len(self.ids)} examples')
        # Different modes of the training defaulted to 'normal'
        if mode == 'augmentation':
            self.augmentation_pipeline = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ],p=1)

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale, round):
        w, h = pil_img.size
        # resizes the image based on scaling factor
        newW, newH = int(scale * w), int(scale * h) 
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255
        
        # if the scaling factor is too small the mask might lose the binary 
        # nature of it's values. We make sure in occasions of extreme scaling
        # the mask still would stay binary by rounding the resized mask values to 1 or 0
        if round:
            img_trans = np.around(img_trans)

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')
        
#         assert len(mask_file) == 1, \
#             f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        
        mask = torch.load(mask_file[0])
        img = torch.load(img_file[0])

        assert img.shape == mask.shape, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'
        
        # In case we only have 1 input image for normal and augmentation mode
        if self.mode == 'augmentation' or self.mode == 'normal':
            # Augments the data for the training dataset
            if self.tag == 'train' and self.mode == 'augmentation':
                augmented = self.augmentation_pipeline(image = np.array(img), mask = np.array(mask))
                img = Image.fromarray(augmented['image'])
                mask = Image.fromarray(augmented['mask'])
            # Runs for validation phase of augmentation mode, 
            # and training and validation phase of normal mode             
            output_img = img
            output_mask = mask
   

        return {
            'image': output_img,
            'mask': output_mask}