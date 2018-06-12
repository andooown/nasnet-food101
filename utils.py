# -*- coding: utf-8 -*-

from keras.applications.nasnet import preprocess_input
from keras.preprocessing import image
from PIL import Image
import numpy as np
import os


def check_directory(dir_path):
    if not os.path.exists(dir_path):
        print('Create directory: {}'.format(dir_path))
        os.makedirs(dir_path, exist_ok=True)


def preprocess(img_file, target_size, transform=True):
    img = Image.open(img_file)
    w, h = img.size

    if transform:
        if w > h:
            ratio = (min(target_size) * 1.2) / float(h)
        else:
            ratio = (min(target_size) * 1.2) / float(w)
        output = img.resize((int(w * ratio), int(h * ratio)))
        output = output.convert("RGB")
        output = image.img_to_array(output)

        # Random rotation
        output = image.random_rotation(output, 30, row_axis = 0, col_axis = 1, channel_axis = 2, fill_mode = 'nearest')
        # Random crop
        left = np.random.randint(0, output.shape[1] - target_size[0] + 1)
        upper = np.random.randint(0, output.shape[0] - target_size[1] + 1)
        output = output[upper:upper + target_size[1], left:left + target_size[0]]

        # Random horizontal flip
        if np.random.rand() < 0.5:
            output = output[:, ::-1, :]
    else:
        if w > h:
            left = (w - h) // 2
            output = img.crop((left, 0, left + h, h))
        else:
            upper = (h - w) // 2
            output = img.crop((0, upper, w, upper + w))
        output = output.resize(target_size)
        output = output.convert("RGB")
        output = image.img_to_array(output)

    output = preprocess_input(output)

    return output
