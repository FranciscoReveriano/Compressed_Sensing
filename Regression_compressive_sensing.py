# This is for ECE580: Intro to machine learning Spring 2020 in Duke
# This is translated to Python from show_chanWeights.m file provided by Prof. Li by 580 TAs

# import ext libs
import numpy as np
import math
import torch
import matplotlib.pyplot as plt
from DCT.DCT_Matrix import DCT_Matrix
from util.util import *
from sklearn.feature_extraction import image
from MOSEK.mosek import *
from tqdm import tqdm


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

def test_first_Patch():
    # Prepare CUDA
    if torch.cuda.is_available():
        device = torch.device("cuda")

    boat = "fishing_boat.bmp"
    lena = "lena.bmp"

    # Read the Boat Images
    matrix = imgRead(boat)                                                                                                  # Read Image into Matrix
    P, Q = matrix.shape                                                                                                     # Width and Length of Matrix
    print("Original Image")
    print("X:", P, "Y:",Q)

    # Create Transformation matrix
    dimension = (4,4)                                                                                                       # Dimension for Block
    T_Matrix = DCT_Matrix(dimension[0], dimension[1])
    print("Transformation matrix:",T_Matrix.shape)

    # Split the Image Into Patches
    patches_original_image = image.extract_patches_2d(matrix,dimension)                                                     # Turn into Patches the main Matrix
    print("Patch Original Shape:",patches_original_image[0].shape)

    # Test With First Patch
    c = patches_original_image[0].flatten()
    c2 = torch.tensor(c).to(device)
    print(type(c2))
    print("First C Flatten Shape:", c.shape)                                                                                # Dimension Shape should be Dim[0] * Dim[1]

    # Make the Mask Matrix
    ## Using Random Number Generator to Create Mask to Smaple
    mask = np.random.randint(2, size=c.shape)
    mask2 = torch.tensor(mask).to(device)
    print("Mask Shape:",mask.shape)
    print("Mask Non-Sparse Values:", count_non_sparse_values(mask))

    # Get the New Sparse matrix
    new_matrix = mask * c                                                                                                   # Taking Dot Product
    new_matrix2 = mask2 * c2
    ## The Next Part is minimizing this matrix
    C_values = count_non_sparse_values(new_matrix)                                                                          # Count the Values in the C Matrix that are non-sparse
    C_values2 = count_non_sparse_values(new_matrix2)
    print("C Non-Sparse Values:", C_values)
    ## Next part is creating the new sparse C matrix
    B_Matrix = convert_C_to_B(C_values, new_matrix)
    B_Matrix2 = convert_C_to_B(C_values2, new_matrix2)
    B_values = count_non_sparse_values(B_Matrix)
    B_values2 = count_non_sparse_values(B_Matrix2)
    print("B Non-Sparse Values:", B_values)
    assert(B_values == B_Matrix.shape[0])

    # Convert the T Matrix to A
    A_Matrix = convert_T_to_A(mask, T_Matrix)
    A_values = count_non_sparse_values(A_Matrix.T[0])
    print("A Non-Sparse Values:", A_values)

    # Need to Random Initialize Alpha
    alpha = np.random.random_sample(T_Matrix[1].shape)                                                                      # Alpha is randomly initiliazed

    # At this point we need to set up our optimizer
    print("B Shape:", B_Matrix.shape)
    print("A Shape:", A_Matrix.shape)
    print("Alpha Shape:", alpha.shape)

    alpha, res = l1norm(A_Matrix,B_Matrix)
    C = np.matmul(T_Matrix,alpha)
    C = C.reshape((dimension))

    print(patches_original_image[0])
    print(C)

def test_whole_image():
    # Prepare CUDA
    if torch.cuda.is_available():
        device = torch.device("cuda")

    dimension = (16, 16)  # Dimension for Block
    boat = "fishing_boat.bmp"
    lena = "lena.bmp"

    # Read the Boat Images
    matrix = imgRead(boat)  # Read Image into Matrix
    P, Q = matrix.shape  # Width and Length of Matrix
    print(P,Q)

    # Create Transformation matrix
    T_Matrix = torch.tensor(DCT_Matrix(dimension[0], dimension[1])).to(device)

    # Split the Image Into Patches
    patches = image.extract_patches_2d(matrix, dimension)  # Turn into Patches the main Matrix
    print(patches.shape)
    #print(type(patches))
    #print(type(patches[0]))
    print(patches[0])
    # Random Initilize Mask
    ## Using Random Number Generator to Create Mask to Smaple
    mask = torch.tensor(np.random.randint(2, size=patches[0].flatten().shape)).to(device)

    # Transform the Patches
    new_image = []
    for patch in tqdm(patches):
        patch = torch.tensor(patch).to(device)
        new_patch = transform_Patch(dimension,mask, patch, T_Matrix)
        new_image.append(new_patch)
    new_image = np.asarray(new_image)
    print("")
    print(new_image[0])
    print(new_image.shape)
    reconstructed_image = image.reconstruct_from_patches_2d(new_image,image_size=(P,Q))
    #plt.imshow(reconstructed_image)
    f = plt.figure()
    ax1 = f.add_subplot(1,2,1)
    plt.imshow(matrix)
    ax1.set_title("Original")
    ax2 =f.add_subplot(1,2,2)
    plt.imshow(reconstructed_image)
    ax2.set_title("Reconstructed")
    plt.title("Block 16 x 16")
    plt.savefig("Block16x16.png")
    plt.show()

test_whole_image()
