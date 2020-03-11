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
from sklearn.metrics import mean_squared_error
from scipy.signal import medfilt2d


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

def test_whole_image(imgIn, blkSize, numSample, filter=False):
    # Make Sure That We are Not Trying to Sample More Points than the Size of the Mask
    assert(numSample < blkSize*blkSize)

    # Set the Desired Dimensions
    dimension = (blkSize, blkSize)                                                                                      #(8, 8)  # Dimension for Block

    # Read the Boat Images
    matrix = imgRead(imgIn)                                                                                             # Read Image into Matrix
    P, Q = matrix.shape                                                                                                 # Width and Length of Matrix

    # Create Transformation matrix
    T_Matrix = DCT_Matrix(dimension[0], dimension[1])

    # Split the Image Into Patches
    patches = image.extract_patches_2d(matrix, dimension)  # Turn into Patches the main Matrix

    # Random Initilize Mask
    mask = create_mask(numSample, dimension[0] * dimension[1])  # New Mask is Made in Each Iteration

    # Transform the Patches
    new_image = []
    # MSE LIST
    MSE_List = []
    for patch in tqdm(patches):
        # Proceed To Do Function On Each Patch
        new_patch = transform_Patch(dimension,mask, patch, T_Matrix)
        # Apply Median Filter
        if filter == True:
            new_patch = medfilt2d(new_patch,kernel_size=3)
        # Append Patch To Image to Recreate Image
        new_image.append(new_patch)
        # Calculate MSE Square
        MSE =mean_squared_error(patch, new_patch)
        MSE_List.append(MSE)

    # Convert List to Numpy Matrix in Desired Dimensions
    new_image = np.asarray(new_image)
    # print("")
    # print(new_image[0])
    # print(new_image.shape)

    # Calculate MSE In Each Image
    MSE_List = np.asarray(MSE_List)
    #print("MSE Average:", np.average(MSE_List))

    # Use SciKit To Reconstruct the Image
    reconstructed_image = image.reconstruct_from_patches_2d(new_image,image_size=(P,Q))
    # Image MSE
    Image_MSE = mean_squared_error(matrix, reconstructed_image)
    #print("MSE Total:", Image_MSE)

    #plt.imshow(reconstructed_image)
    #fig, (ax_1, ax_2) = plt.subplots(nrows=1, ncols=2,sharex=True)
    #ax_1.set_title("Original Image")
    #ax_1.imshow(matrix)
    #ax_2.set_title("Reconstructed Image")
    #ax_2.imshow(reconstructed_image)
    #plt.savefig("Block16x16.png")
    #title = "Block Size = " + str(blkSize) + " x " + str(blkSize) + " & Mask=" + str(numSample)
    #fig.suptitle(title)
    #plt.show()
    return reconstructed_image, Image_MSE

def imgRecover(imgIn, blkSize, numSample):
    """
    Recover the input image from a small size samples
    :param imgIn: input image
    :param blkSize: block size
    :param numSample: how many samples in each block
    :return: recovered image
    """
    ##### Your Implementation here
    print("Size of Block:",blkSize)
    print("Mask Sample:",numSample)
    test_whole_image(imgIn, blkSize, numSample)
    return None

def main_boat_8x8():
    boat = "/home/franciscoAML/Documents/Compressed_Sensing/fishing_boat.bmp"
    print("Boat Information")
    print("Block Size 8 x 8 ")
    ###########################################################################
    print("Mask = 10")
    reconstructed_img10, MSE_10 = test_whole_image(boat,8,10)
    print("Mask=10, MSE=",MSE_10)
    ###########################################################################
    print("Mask = 20")
    reconstructed_img20, MSE_20 = test_whole_image(boat,8,20)
    print("Mask=20, MSE=", MSE_20)
    ###########################################################################
    print("Mask = 30")
    reconstructed_img30, MSE_30 = test_whole_image(boat,8,30)
    print("Mask=30, MSE=", MSE_30)
    ###########################################################################
    print("Mask = 40")
    reconstructed_img40, MSE_40 = test_whole_image(boat,8,40)
    print("Mask=40, MSE=", MSE_40)
    ###########################################################################
    print("Mask = 50")
    reconstructed_img50, MSE_50 = test_whole_image(boat,8,50)
    print("Mask=50, MSE=", MSE_50)
    ###########################################################################
    ### Proceed to Graph#######################################################
    print("Graphing Results")
    fig, (ax_1, ax_2, ax_3, ax_4, ax_5, ax_6) = plt.subplots(nrows=2, ncols=6, sharex=True)
    # Original Image
    ax_1.set_title("Original Image")
    ax_1.imshow(imgRead(boat))
    # Sample = 10
    ax_2.set_title("Sample = 10")
    ax_2.imshow(reconstructed_img10)
    # Sample = 20
    ax_3.set_title("Sample = 20")
    ax_3.imshow(reconstructed_img20)
    # Sample = 30
    ax_4.set_title("Sample = 30")
    ax_4.imshow(reconstructed_img30)
    # Sample = 40
    ax_5.set_title("Sample = 40")
    ax_5.imshow(reconstructed_img40)
    # Sample 50
    ax_6.set_title("Sample = 50")
    ax_6.imshow(reconstructed_img50)
    title = "Boat: (Block Size = 8 x 8) & (No Filtering)"
    fig.suptitle(title)
    plt.savefig("Boat8x8.png")
    #plt.show()

main_boat_8x8()