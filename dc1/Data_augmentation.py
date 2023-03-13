# importing all the required libraries
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import skimage.io as io
from skimage.transform import rotate, AffineTransform, warp
from skimage.util import random_noise
from skimage.filters import gaussian
import matplotlib.pyplot as plt
from dc1.image_dataset import ImageDataset

# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary  # type: ignore
# poopy


# Other imports
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.pyplot import figure
import os
import argparse
import plotext  # type: ignore
from datetime import datetime
from pathlib import Path
from typing import List

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

# Loading the training datasets
data_train_X = np.load('C:\GitHub\JBG040-Group10\data\X_train.npy')
data_train_Y = np.load('C:\GitHub\JBG040-Group10\data\X_train.npy')

# Loading the testing datasets
data_test_X = np.load('C:\GitHub\JBG040-Group10\data\X_test.npy')
data_test_Y = np.load('C:\GitHub\JBG040-Group10\data\Y_test.npy')

# Left/right/up/down translatiion
def translate(img, shift=10, direction='left', roll=True):
    assert direction in ['right', 'left', 'down', 'up'], 'Directions should be top|up|left|right'
    img = img.copy()
    if direction == 'right':
        right_slice = img[:, -shift:].copy()
        img[:, shift:] = img[:, :-shift]
        if roll:
            img[:, :shift] = np.fliplr(right_slice)
    if direction == 'left':
        left_slice = img[:, :shift].copy()
        img[:, :-shift] = img[:, shift:]
        if roll:
            img[:, -shift:] = left_slice
    if direction == 'down':
        down_slice = img[-shift:, :].copy()
        img[shift:, :] = img[:-shift, :]
        if roll:
            img[:shift, :] = down_slice
    if direction == 'up':
        upper_slice = img[:shift, :].copy()
        img[:-shift, :] = img[shift:, :]
        if roll:
            img[-shift:, :] = upper_slice
    return img

img = data_test_X[0][0]
print(translate(img, direction='up', shift=20))
print(translate(img, direction='down', shift=20))
