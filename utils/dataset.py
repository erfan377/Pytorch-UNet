from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import albumentations as A


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, mode='normal'):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.mode = mode
        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')
        if mode == 'augmentation':
            self.augmentation_pipeline = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ],p=1)
        if mode == 'temporal_augmentation':
            self.augmentation_pipeline = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ],
            additional_targets={'img2': 'image', 'img3': 'image', 'img4': 'image'})
            

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale, round):
        w, h = pil_img.size
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
            
        if round:
            img_trans = np.around(img_trans)

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'
            
        if self.mode == 'temporal' or self.mode == 'temporal_augmentation':
            img_file2 = glob('data/imgs_jan/' + idx + '.*')
            img_file3 = glob('data/imgs_apr/' + idx + '.*')
            img_file4 = glob('data/imgs_oct/' + idx + '.*')
            img2 = Image.open(img_file2[0])
            img3 = Image.open(img_file3[0])
            img4 = Image.open(img_file4[0])
        
        if self.mode == 'augmentation' or self.mode == 'normal':
            if self.tag == 'train' and self.mode == 'augmentation':
                augmented = self.augmentation_pipeline(image = np.array(img), mask = np.array(mask))
                img = Image.fromarray(augmented['image'])
                mask = Image.fromarray(augmented['mask'])

            output_img = self.preprocess(img, self.scale, False)
            output_mask = self.preprocess(mask, self.scale, True)
        
        if self.mode == 'temporal_augmentation' or self.mode == 'temporal':
            if self.tag == 'train' and self.mode == 'temporal_augmentation':
                augmented = self.augmentation_pipeline(image = np.array(img), img2 = np.array(img2), img3 = np.array(img3), img4 = np.array(img4), mask = np.array(mask))
                img = Image.fromarray(augmented['image'])
                img2 = Image.fromarray(augmented['img2'])
                img3 = Image.fromarray(augmented['img3'])
                img4 = Image.fromarray(augmented['img4'])
                mask = Image.fromarray(augmented['mask'])
            
            img = self.preprocess(img, self.scale, False)
            img2 = self.preprocess(img2, self.scale, False)
            img3 = self.preprocess(img3, self.scale, False)
            img4 = self.preprocess(img4, self.scale, False)
            output_mask = self.preprocess(mask, self.scale, True)                 
            
            output_img = np.vstack((img2, img3, img, img4))

        return {
            'image': torch.from_numpy(output_img).type(torch.FloatTensor),
            'mask': torch.from_numpy(output_mask).type(torch.FloatTensor)
        }
