# importing all the required libraries
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from PIL import Image
from PIL.ImageTransform import AffineTransform

# Torch imports
from torchsummary import summary  # type: ignore

# Other imports
from skimage.transform import warp
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.pyplot import figure
import os
import argparse
import plotext  # type: ignore

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

# Loading the training datasets
data_train_X = np.load('C:\GitHub\JBG040-Group10\data\X_train.npy')
data_train_Y = np.load('C:\GitHub\JBG040-Group10\data\Y_train.npy')

# Loading the testing datasets
data_test_X = np.load('C:\GitHub\JBG040-Group10\data\X_test.npy')
data_test_Y = np.load('C:\GitHub\JBG040-Group10\data\Y_test.npy')


# Selecting the first image from test data for X
img_numpy = data_test_X[0][0]
image = Image.fromarray(img_numpy)

# Creating the random shifting transformation
transform = transforms.RandomAffine(degrees=0, translate=(0.2, 0), fill=0)

# Apply the transformation to the image
shifted_img = transform(image)
shifted_img.show()

#shit