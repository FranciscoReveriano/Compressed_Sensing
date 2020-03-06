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

def DCTMATRIX(P,Q):
    # First Create Empty Zero NP Matrix
    DCT = np.zeros((P*Q, P*Q))  # DCT Matrix
    print("DCT Shape:", DCT.shape)
    for x in range(DCT.shape[0]):
        for y in range(DCT.shape[1]):
            # This is just to make the following of the math formula easier
            u = x
            v = y
            # Calculate A
            if u == 1:
                a = math.sqrt(1 / P)
            else:
                a = math.sqrt(2 / P)
            # Calculate B
            if v == 1:
                b = math.sqrt(1 / Q)
            else:
                b = math.sqrt(2 / Q)
            # Make the G Calculation
            DCT[x][y] = a * b * math.cos((math.pi * (2 * x + 1) * (u - 1)) / (2 * P)) * math.cos(
                (math.pi * (2 * y - 1) * (v - 1)) / (2 * Q))  # Calculate the Matrix
    print(DCT[0])
    return DCT


boat = "fishing_boat.bmp"
lena = "lena.bmp"
dimension = (8,8)

# Read the Images
matrix = imgRead(boat)                                                                                                  # Read Image into Matrix
P, Q = matrix.shape                                                                                                     # Width and Length of Matrix
print("X:", P, "Y:",Q)

# Create the DCT Matrix
T_Matrix = DCTMATRIX(8,8)                                                                                               # Matrix is Divided into 8 x 8 Parts
print("T Shape:", T_Matrix.shape)

# Break Image Into Patches
patches_original_image = image.extract_patches_2d(matrix,dimension)                                                     # Turn into Patches the main Matrix

# Utilize Patch 1
patch_1 = patches_original_image[0]
patch_1 = patch_1.flatten()

