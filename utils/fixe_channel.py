'''
This file is for to preprocess images from Planet Lab
to convert them from 4 bands to 3 bands and crop them to
a fixed size.
'''

import json
import numpy as np
import sys
import csv
import matplotlib.pyplot as plt
from PIL import Image
import os
from collections import defaultdict 

base_dir = 'data/'
imgs_filenames = [f for f in os.listdir(base_dir + 'imgs_planet/')]
cut_size = 467

for img_file in imgs_filenames:
        image_id = img_file.split('.')[0] #get image ID
        im_name = str(int(image_id)) +'.tif'
        rgba_image = Image.open(base_dir + 'imgs/' + im_name)
        #turn to RGB
        rgba_image.load()
        background = Image.new("RGB", rgba_image.size, (255, 255, 255))
        background.paste(rgba_image, mask = rgba_image.split()[3])
        #crop to a unanimous size
        background = background.crop((0, 0, cut_size, cut_size)) 
        #size image
        overlay_path = base_dir + 'imgs/' + str(int(image_id)) + '.jpeg'
        background.save(overlay_path, "JPEG", quality=100)