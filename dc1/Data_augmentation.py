# importing all the required libraries
import warnings
warnings.filterwarnings('ignore')

import random


import numpy as np
from PIL import Image
from PIL.ImageTransform import AffineTransform

# Torch imports
import torch
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
# since the data_test_X is a numpy.nd array of only one column, the random function will pick values for which row (picture) to augment

# create a loop and put everything down below in the loop; inlcude random number generator for the images and
img_numpy = data_test_X[9][0] # change to data_train_X!!!!
image = Image.fromarray(img_numpy)
image.show()

# generating a random number to use as input for random adjust sharpener
sharpness_factor = random.sample(range(0,3),1)[0]
# print(sharpness_factor)

# Creating the brightness transformation
transform = transforms.Compose([
    transforms.ColorJitter(brightness=(0.5,1.5)),
    transforms.RandomAffine(degrees=0, translate=(0.2, 0), fill=0),
    transforms.RandomAdjustSharpness(sharpness_factor=sharpness_factor,p=0.3),
    transforms.RandomRotation(degrees=(0,10))
   ])

# Apply the transformation to the image
shifted_img = transform(image)
# shifted_img.show()

# create an empty numpy array to store augmented images
augmented_images = np.empty_like(data_test_X) # remember to change this to data_train_X

# iterate over the original images, apply the transformation pipeline and store the augmented images
for idx, image in enumerate(data_test_X): # remember to change this to data_train_X
    augmented_image = transform(torch.tensor(image)).numpy()
    augmented_images[idx] = augmented_image

# concatenate the original and augmented images along the first axis
all_images = np.concatenate((data_test_X, augmented_images), axis=0)

print(len(all_images))
for i in all_images:
    print(i)

testing = Image.fromarray(all_images[9][0])
testing.show()

