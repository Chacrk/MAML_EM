
from __future__ import print_function
import csv
import glob
import os

from PIL import Image

path_to_images = '../data/miniimagenet/images/'

all_images = glob.glob(path_to_images + '*')

# Resize images
for i, image_file in enumerate(all_images):
    im = Image.open(image_file)
    im = im.resize((80, 80), resample=Image.LANCZOS)
    im.save(image_file)
    if i % 500 == 0:
        print(i)

