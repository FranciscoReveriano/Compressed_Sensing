# This is for ECE580: Intro to machine learning Spring 2020 in Duke
# This is translated to Python from show_chanWeights.m file provided by Prof. Li by 580 TAs

# import ext libs
import numpy as np
import math
import matplotlib.pyplot as plt
from DCT.DCT_Matrix import DCT_Matrix
from util.util import *
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

# Read the Boat Images
matrix = imgRead(boat)                                                                                                  # Read Image into Matrix
P, Q = matrix.shape                                                                                                     # Width and Length of Matrix
print("Original Image")
print("X:", P, "Y:",Q)

# Create Transformation matrix
dimension = (3,3)                                                                                                       # Dimension for Block
T_Matrix = DCT_Matrix(dimension[0], dimension[1])
print("Transformation matrix:",T_Matrix.shape)

# Split the Image Into Patches
patches_original_image = image.extract_patches_2d(matrix,dimension)                                                     # Turn into Patches the main Matrix
print("Patch Original Shape:",patches_original_image[0].shape)

# Test With First Patch
c = patches_original_image[0].flatten()
print("First C Flatten Shape:", c.shape)                                                                                # Dimension Shape should be Dim[0] * Dim[1]

# Make the Mask Matrix
## Using Random Number Generator to Create Mask to Smaple
mask = np.random.randint(2, size=c.shape)
print("Mask:", mask)
print("Mask Shape:",mask.shape)
print("Mask Non-Sparse Values:", count_non_sparse_values(mask))

# Get the New Sparse matrix
new_matrix = mask * c                                                                                                   # Taking Dot Product
## The Next Part is minimizing this matrix
C_values = count_non_sparse_values(new_matrix)                                                                          # Count the Values in the C Matrix that are non-sparse
print("C Non-Sparse Values:", C_values)
## Next part is creating the new sparse C matrix
B_Matrix = convert_C_to_B(C_values, new_matrix)
B_values = count_non_sparse_values(B_Matrix)
print("B Non-Sparse Values:", B_values)
assert(B_values == B_Matrix.shape[0])

# Convert the T Matrix to A


# Now we Need to Convert the T Matrix into Smaller A Matrix
A_matrix = (T_Matrix.T * mask).T                                                                                        # Multiply the matrix by mask to make sparse
# We Shrink the Matrix
A_Values = count_non_sparse_values(A_matrix.T[0])
print("A: Matrix\n", A_matrix)
print("T Non-Sparse Values:",A_Values)
## Now I need to Convert To Smaller Matrix
A_matrix_small = np.zeros((A_Values,T_Matrix.shape[0]))
print(A_matrix_small.shape)

A = A_matrix[A_matrix != 0]
print(A.reshape(A_matrix_small.shape))