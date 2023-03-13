# importing all the required libraries
import warnings
warnings.filterwarnings('ignore')

import random


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
data_train_X = np.load(r'C:\Users\ishik\OneDrive\Documents\GitHub\Vis final project\DataChallenge1\data\X_train.npy')
data_train_Y = np.load(r'C:\Users\ishik\OneDrive\Documents\GitHub\Vis final project\DataChallenge1\data\Y_train.npy')

# Loading the testing datasets
data_test_X = np.load(r'C:\Users\ishik\OneDrive\Documents\GitHub\Vis final project\DataChallenge1\data\X_test.npy')
data_test_Y = np.load(r'C:\Users\ishik\OneDrive\Documents\GitHub\Vis final project\DataChallenge1\data\X_train.npy')


# Selecting the first image from test data for X
img_numpy = data_test_X[1][0]
image = Image.fromarray(img_numpy)
image.show()

# generating a random number to use as input for random adjust sharpener
sharpness_factor = random.sample(range(0,3),1)[0]
print(sharpness_factor)

# Creating the brightness transformation
transform = transforms.Compose([
    transforms.ColorJitter(brightness=(0.5,1.5)),
    transforms.RandomAffine(degrees=0, translate=(0.2, 0), fill=0),
    transforms.RandomAdjustSharpness(sharpness_factor=sharpness_factor,p=0.3),
    transforms.RandomRotation(degrees=(0,10))
   ])

# Apply the transformation to the image
shifted_img = transform(image)
shifted_img.show()

