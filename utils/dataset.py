#########################################
# Dataset loader class from the folders #
#########################################

from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import albumentations as A
import pdb


class BasicDataset(Dataset):
    def __init__(self, imgs_list, imgs_dir, masks_dir, epochs, scale=1, tag='train', mode='normal'):
        print('hiiiiiiii')
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
#         if mode == 'temporal_augmentation':
#             self.augmentation_pipeline = A.Compose([
#                 A.HorizontalFlip(p=0.5),
#                 A.VerticalFlip(p=0.5),
#                 A.RandomRotate90(p=0.5),
#             ],
#             additional_targets={'img2': 'image', 'img3': 'image', 'img4': 'image'})
            

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
#         import pdb;pdb.set_trace()
        mask_file = glob(self.masks_dir + idx + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')
#         print('maks', mask_file)
#         print('img', img_file)
        
#         assert len(mask_file) == 1, \
#             f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        
        img = torch.load(img_file[0])
        if len(mask_file) == 0:
            mask = torch.zeros_like(img)
        else:
            mask = torch.load(mask_file[0])

#         print('img size', img.shape)
#         print('mask size', mask.shape)
        assert img.shape == mask.shape, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'
            
#         # Read additional images incase we train on temporal mode
#         if self.mode == 'temporal' or self.mode == 'temporal_augmentation':
#             img_file2 = glob('data/imgs_jan/' + idx + '.*')
#             img_file3 = glob('data/imgs_apr/' + idx + '.*')
#             img_file4 = glob('data/imgs_oct/' + idx + '.*')
#             img2 = Image.open(img_file2[0])
#             img3 = Image.open(img_file3[0])
#             img4 = Image.open(img_file4[0])
        
        # In case we only have 1 input image for normal and augmentation mode
        if self.mode == 'augmentation' or self.mode == 'normal':
            # Augments the data for the training dataset
            if self.tag == 'train' and self.mode == 'augmentation':
                augmented = self.augmentation_pipeline(image = np.array(img), mask = np.array(mask))
                img = Image.fromarray(augmented['image'])
                mask = Image.fromarray(augmented['mask'])
            # Runs for validation phase of augmentation mode, 
            # and training and validation phase of normal mode             
#             output_img = self.preprocess(img, self.scale, False)
#             output_mask = self.preprocess(mask, self.scale, True)
#               output_img = img
#               output_mask = mask
#         # When training happens on temporal mode
#         if self.mode == 'temporal_augmentation' or self.mode == 'temporal':
#             # Augments the data for the training dataset
#             if self.tag == 'train' and self.mode == 'temporal_augmentation':
#                 augmented = self.augmentation_pipeline(image = np.array(img), img2 = np.array(img2), img3 = np.array(img3), img4 = np.array(img4), mask = np.array(mask))
#                 img = Image.fromarray(augmented['image'])
#                 img2 = Image.fromarray(augmented['img2'])
#                 img3 = Image.fromarray(augmented['img3'])
#                 img4 = Image.fromarray(augmented['img4'])
#                 mask = Image.fromarray(augmented['mask'])
#             # Runs for validation phase of augmentation mode, 
#             # and training and validation phase of normal mode
#             img = self.preprocess(img, self.scale, False)
#             img2 = self.preprocess(img2, self.scale, False)
#             img3 = self.preprocess(img3, self.scale, False)
#             img4 = self.preprocess(img4, self.scale, False)
#             output_mask = self.preprocess(mask, self.scale, True)                 
            
#             # Stack images on top. Nor ordered based on the
#             # progression of seasons during the year
#             output_img = np.vstack((img2, img3, img, img4)) 

        return {
            'image': img,
            'mask': mask
        }
