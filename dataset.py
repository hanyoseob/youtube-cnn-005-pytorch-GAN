import os
import numpy as np

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from util import *
from skimage.transform import rescale, resize

## 데이터 로더를 구현하기
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, task=None, opts=None):
        self.data_dir = data_dir
        self.transform = transform
        self.task = task
        self.opts = opts

        # Updated at Apr 5 2020
        self.to_tensor = ToTensor()

        lst_data = os.listdir(self.data_dir)
        lst_data = [f for f in lst_data if f.endswith('jpg') | f.endswith('jpeg') | f.endswith('png')]
        lst_data.sort()

        # lst_data = np.load('/content/drive/My Drive/YouTube/youtube-cnn-005-pytorch-GAN/lst_data.npy')

        self.lst_data = lst_data

    def __len__(self):
        return len(self.lst_data)

    def __getitem__(self, index):
        img = plt.imread(os.path.join(self.data_dir, self.lst_data[index]))
        sz = img.shape

        """
        2020.04.19. Edited by YS
        """
        # if sz[1] > sz[0]:
        #     img = img.transpose((1, 0, 2))

        if img.ndim == 2:
            img = img[:, :, np.newaxis]

        if img.dtype == np.uint8:
            img = img / 255.0

        # Updated at Apr 5 2020
        data = {'label': img}

        if self.task == "inpainting":
            data['input'] = add_sampling(data['label'], type=self.opts[0], opts=self.opts[1])
        elif self.task == "denoising":
            data['input'] = add_noise(data['label'], type=self.opts[0], opts=self.opts[1])

        if self.transform:
            data = self.transform(data)

        if self.task == "super_resolution":
            data['input'] = add_blur(data['label'], type=self.opts[0], opts=self.opts[1])

        data = self.to_tensor(data)

        return data

## 트렌스폼 구현하기
class ToTensor(object):
    def __call__(self, data):
        # Updated at Apr 5 2020
        for key, value in data.items():
            value = value.transpose((2, 0, 1)).astype(np.float32)
            data[key] = torch.from_numpy(value)

        return data

class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        # Updated at Apr 5 2020
        for key, value in data.items():
            data[key] = (value - self.mean) / self.std

        return data


class RandomFlip(object):
    def __call__(self, data):

        if np.random.rand() > 0.5:
            # Updated at Apr 5 2020
            for key, value in data.items():
                data[key] = np.flip(value, axis=0)

        if np.random.rand() > 0.5:
            # Updated at Apr 5 2020
            for key, value in data.items():
                data[key] = np.flip(value, axis=1)

        return data


class RandomCrop(object):
  def __init__(self, shape):
      self.shape = shape

  def __call__(self, data):
    h, w = data['label'].shape[:2]
    new_h, new_w = self.shape

    top = np.random.randint(0, h - new_h)
    left = np.random.randint(0, w - new_w)

    id_y = np.arange(top, top + new_h, 1)[:, np.newaxis]
    id_x = np.arange(left, left + new_w, 1)

    # Updated at Apr 5 2020
    for key, value in data.items():
        data[key] = value[id_y, id_x]

    return data


"""
2020.04.19. Edited by YS
"""
class Resize(object):
  def __init__(self, shape):
      self.shape = shape

  def __call__(self, data):
    # Updated at Apr 5 2020
    for key, value in data.items():
        data[key] = resize(value, output_shape=(self.shape[0], self.shape[1], value.shape[2]))

    return data