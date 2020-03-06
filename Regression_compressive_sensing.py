# This is for ECE580: Intro to machine learning Spring 2020 in Duke
# This is translated to Python from show_chanWeights.m file provided by Prof. Li by 580 TAs

# import ext libs
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import fft, dct
import cv2
from sklearn.feature_extraction import image
#from scipy.misc import imread   # Make Sure you install the required packages like Pillow and scipy

def imgRead(fileName):
    """
    load the input image into a matrix
    :param fileName: name of the input file
    :return: a matrix of the input image
    Examples: imgIn = imgRead('lena.bmp')
    """
    imgIn = plt.imread(fileName)
    return imgIn

def imgShow(imgOut):
    """
    show the image saved in a matrix
    :param imgOut: a matrix containing the image to show
    :return: None
    """
    imgOut = np.uint8(imgOut)
    plt.imshow(imgOut)
    plt.show()

def imgRecover(imgIn, blkSize, numSample):
    """
    Recover the input image from a small size samples
    :param imgIn: input image
    :param blkSize: block size
    :param numSample: how many samples in each block
    :return: recovered image
    """
    ##### Your Implementation here

    return None


boat = "fishing_boat.bmp"
lena = "lena.bmp"
dimension = (8,8)

# Read the Images
matrix = imgRead(boat)                                                                                                  # Read Image into Matrix
P, Q = matrix.shape                                                                                                     # Width and Length of Matrix
print("X:", P, "Y:",Q)
## Create the DCT Matrix
DCT = np.zeros((P,Q))                                                                                                   # DCT Matrix
# First Create Empty Zero NP Matrix
for x in range(P):
    for y in range(Q):
        # This is just to make the following of the math formula easier
        u = x
        v = y
        # Calculate A
        if u == 1:
            a = math.sqrt(1/P)
        else:
            a = math.sqrt(2/P)
        # Calculate B
        if v == 1:
            b = math.sqrt(1/Q)
        else:
            b = math.sqrt(2/Q)
        # Make the G Calculation
        DCT[x][y] = a * b * math.cos((math.pi * (2*x + 1)*(u-1))/(2*P)) * math.cos((math.pi * (2* y - 1) * (v-1))/(2*Q))# Calculate the Matrix


# Create Flat Original Image Matrix
flat_original_matrix = matrix.flatten()
assert(P*Q == flat_original_matrix.shape[0])                                                                            # Make sure that the matrix shape is correct
print("Flat Matrix Shape:", flat_original_matrix.shape)

# patches_original_image = image.extract_patches_2d(matrix,dimension)                                                     # Turn into Patches the main Matrix
# patches_dct_matrix = image.extract_patches_2d(DCT, dimension)                                                           # Turn into Patches the DCT Matrix

# Make the Mask Matrix
## Using Random Number Generator to Create Mask to Smaple
mask = np.random.randint(2,size=flat_original_matrix.shape)

# Get the New Sparse matrix
new_matrix = mask * flat_original_matrix                                                                                # Taking Dot Product
## The Next Part is minimizing this matrix
values = 0                                                                                                              # Size of Non-Discrete Values
for i in range(len(new_matrix)):                                                                                        # This part is to calculate how many values are not null values
    if new_matrix[i] != 0:
        values += 1
B_Matrix = np.zeros((values))
assert(len(B_Matrix) == values)                                                                                         # Make sure the matrix size is correct to fit the values
j = 0                                                                                                                    # Index Number of B Matrix
for i in range(len(new_matrix)):                                                                                        # Go Through the C Matrix
    if new_matrix[i] != 0:                                                                                              # if value is not discrete we put it into the new B
        B_Matrix[j] = new_matrix[i]
        j += 1

# Now we Need to Arrange the new smaller A Matrix
new_matrix = mask * DCT