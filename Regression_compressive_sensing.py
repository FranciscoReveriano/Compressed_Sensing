
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
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from scipy.signal import medfilt2d
from statistics import mean

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


def test_whole_image_KFOLD(imgIn, blkSize, numSample, filter=False, solver="L2", display=False, lambda1=0, K_FOLD=True, Solve=True, fileName="Results.txt",sampleSize=100):
    # Make Sure That We are Not Trying to Sample More Points than the Size of the Mask
    assert(numSample < blkSize*blkSize)

    # Set the Desired Dimensions
    dimension = (blkSize, blkSize)                                                                                      #(8, 8)  # Dimension for Block

    # Read the Boat Images
    matrix = imgRead(imgIn)[:,:,0]                                                                                            # Read Image into Matrix
    P, Q = matrix.shape[0], matrix.shape[1]                                                                             # Width and Length of Matrix

    # Create Transformation matrix
    T_Matrix = DCT_Matrix(dimension[0], dimension[1])

    # Split the Image Into Patches
    patches = image.extract_patches_2d(matrix, dimension)  # Turn into Patches the main Matrix

    # Random Initilize Mask
    mask = create_mask(numSample, dimension[0] * dimension[1])  # New Mask is Made in Each Iteration
    # Conduct K-Folds to find best alpha value
    txtFile = fileName + ".txt"
    if K_FOLD == True and solver=="Lasso":
        ############################# Declare Training Patch ###########################################################
        training_patches = []
        for i in range(sampleSize):
            j = np.random.randint(len(patches))
            training_patches.append(patches[j])
        training_patches = np.asarray(training_patches)
        ############################ Lambda List #######################################################################
        # Declare Lambda Range
        lambdas = [10 ** (-i) for i in range(1, 6)]
        # Begin Optimal Testing
        #print("Conducting K-Fold Testing to Find Optimal Lambda")
        kf = KFold(n_splits=6, shuffle=True)                                                                            # Declare K-Fold
        lambda_list = []                                                                                                # Lambda List
        lambdas_list = []                                                                                               # Lambdas
        index_i = 1                                                                                                     # Index
        ################################## Conduct K-Folds #############################################################
        for i in tqdm(range(20), desc="Run", position=0):
            print("##################### Running Trial", index_i, "######################")  # Print Fold Number
            index_i += 1
            for train_index, test_index in kf.split(training_patches):

                X_train, X_test = training_patches[train_index], training_patches[test_index]
                ################################ Training Fold #############################################################
                # Lambda Error
                lambda_error_list = []
                # Now We Have The Index of the Models We Need To train
                for lambda2 in tqdm(lambdas,desc="Lambda",position=0):
                    # Random Initilize Mask
                    mask = create_mask(numSample, dimension[0] * dimension[1])  # New Mask is Made in Each Iteration
                    MSE_LIST = []
                    for patch in tqdm(X_train,desc="Training",position=1):                                              # Go Through the Training Portion for Lambda
                        local_mse_list = []                                                                             ## Local MSE List to Calculate the 20 Lambda Exam
                        for n in range(1):                                                                              # Run Through Each Loop 20 Times
                            new_patch = transform_Patch(dimension, mask, patch, T_Matrix, solver, lambda1=lambda2)      # Optimize Patch
                            MSE = mean_squared_error(patch, new_patch)                                                  # Calculate MSE
                            local_mse_list.append(MSE)                                                                  # Append to Local MSE LIST
                        MSE_LIST.append(mean(local_mse_list))
                    # Now Convert the MSE_LIST TO NUMPY and calculate Mean
                    mean_mse_list = mean(MSE_LIST)
                    #print("Average of Lambda:", lambda2, "is:", mean_mse_list)
                    lambda_error_list.append(mean_mse_list)
                #print(lambdas)
                #print(lambda_error_list)
                ############################################################################################################
                best_lambda_index = lambda_error_list.index(min(lambda_error_list))
                best_lambda = lambdas[best_lambda_index]
                #print("Best Lambda:",best_lambda)
                test_mse_list = []
                for patch in tqdm(X_test, desc="Testing", position=1):
                    new_patch = transform_Patch(dimension, mask, patch, T_Matrix, solver, lambda1=best_lambda)          # Optimize Patch
                    MSE = mean_squared_error(patch, new_patch)  # Calculate MSE
                    test_mse_list.append(MSE)
                mean_test_mse_list = mean(test_mse_list)
                #print("MSE of Test:", mean_test_mse_list)
                lambdas_list.append(best_lambda)
                lambda_list.append(mean_test_mse_list)
            # Find the Best Overall Lambda
        min_final_mse = min(lambda_list)
        print("Minimum MSE:",min_final_mse)
        final_lambda_index = lambda_list.index(min_final_mse)
        final_lambda = lambdas_list[final_lambda_index]
        print("Best Lambda:",final_lambda)
        print_line = str(final_lambda) + " " + str(min_final_mse) + "\n"
        # Append Final Results to File
        with open(fileName, 'a') as f:
            f.write(print_line)

    if Solve == True:
        print('############################ Solving Final Image Using best Lambda########################################')
        # Random Initilize Mask
        mask = create_mask(numSample, dimension[0] * dimension[1])  # New Mask is Made in Each Iteration
        patches = image.extract_patches_2d(matrix, dimension)  # Turn into Patches the main Matrix
        # Transform the Patches
        new_image = []
        # MSE LIST
        MSE_List = []
        for patch in tqdm(patches):
            # Proceed To Do Function On Each Patch
            if solver == "L1":
                new_patch = transform_Patch(dimension,mask, patch, T_Matrix,solver)
            if solver == "L2":
                new_patch = transform_Patch(dimension,mask,patch,T_Matrix,solver)
            if solver == "Lasso":
                new_patch = transform_Patch(dimension,mask,patch,T_Matrix,solver,lambda1=final_lambda)
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

        if display == True:
            with open(fileName, 'a') as f:
                linePrint = "MSE Total With Original: " + str(Image_MSE) + "\n"
                f.write(linePrint)
            plt.imshow(reconstructed_image)
            fig, (ax_1, ax_2) = plt.subplots(nrows=1, ncols=2,sharex=True)
            ax_1.set_title("Original Image")
            ax_1.imshow(matrix)
            ax_2.set_title("Reconstructed Image")
            ax_2.imshow(reconstructed_image)
            plt.savefig("Block16x16.png")
            title = "Block Size = " + str(blkSize) + " x " + str(blkSize) + " & Mask=" + str(numSample) + " Lambda:" + str(final_lambda)
            fig.suptitle(title)
            plotName = fileName + "png"
            plt.savefig(plotName)
        return reconstructed_image, Image_MSE


def test_whole_image(imgIn, blkSize, numSample, filter=False, solver="L2", display=False,lambda1=0):
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
        if solver == "L1":
            new_patch = transform_Patch(dimension,mask, patch, T_Matrix,solver)
        if solver == "L2":
            new_patch = transform_Patch(dimension,mask,patch,T_Matrix,solver)
        if solver == "Lasso":
            new_patch = transform_Patch(dimension,mask,patch,T_Matrix,solver,lambda1=lambda1)
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
    print("MSE Total:", Image_MSE)
    if display == True:
        plt.imshow(reconstructed_image)
        fig, (ax_1, ax_2) = plt.subplots(nrows=1, ncols=2,sharex=True)
        ax_1.set_title("Original Image")
        ax_1.imshow(matrix)
        ax_2.set_title("Reconstructed Image")
        ax_2.imshow(reconstructed_image)
        plt.savefig("Block16x16.png")
        title = "Block Size = " + str(blkSize) + " x " + str(blkSize) + " & Mask=" + str(numSample)
        fig.suptitle(title)
        plt.show()
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
    boat = "fishing_boat.bmp"
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
    fig, ((ax_1, ax_2, ax_3), (ax_4, ax_5, ax_6)) = plt.subplots(nrows=2, ncols=3, sharex=True)
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

def main_boat_8x8_filtering():
    boat = "fishing_boat.bmp"
    print("Boat Information With Filtering")
    print("Block Size 8 x 8 ")
    ###########################################################################
    print("Mask = 10")
    reconstructed_img10, MSE_10 = test_whole_image(boat,8,10,filter=True, solver="Lasso", display=False, lambda1=0.001)
    print("Mask=10, MSE=",MSE_10)
    ###########################################################################
    print("Mask = 20")
    reconstructed_img20, MSE_20 = test_whole_image(boat,8,20, filter=True, solver="Lasso", display=False, lambda1=0.001)
    print("Mask=20, MSE=", MSE_20)
    ###########################################################################
    print("Mask = 30")
    reconstructed_img30, MSE_30 = test_whole_image(boat,8,30, filter=True, solver="Lasso", display=False, lambda1=0.001)
    print("Mask=30, MSE=", MSE_30)
    ###########################################################################
    print("Mask = 40")
    reconstructed_img40, MSE_40 = test_whole_image(boat,8,40, filter=True, solver="Lasso", display=False, lambda1=0.001)
    print("Mask=40, MSE=", MSE_40)
    ###########################################################################
    print("Mask = 50")
    reconstructed_img50, MSE_50 = test_whole_image(boat,8,50, filter=True, solver="Lasso", display=False, lambda1=0.001)
    print("Mask=50, MSE=", MSE_50)
    ###########################################################################
    ### Proceed to Graph#######################################################
    print("Graphing Results")
    fig, ((ax_1, ax_2, ax_3), (ax_4, ax_5, ax_6)) = plt.subplots(nrows=2, ncols=3, sharex=True)
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
    title = "Boat: (Block Size = 8 x 8) & (Median Filtering)"
    fig.suptitle(title)
    plt.savefig("Boat8x8_Filtering.png")
    #plt.show()


def main_lena_16x16_filtering():
    boat = "fishing_boat.bmp"
    lena = "lena.bmp"
    print("Lena Information With Filtering")
    print("Block Size 16 x 16 ")
    ###########################################################################
    print("Mask = 10")
    title = "Lena_Mask_10.out"
    reconstructed_img10, MSE_10 = test_whole_image(lena,16,10,filter=True, solver="Lasso", display=False, lambda1=0.00001)
    np.savetxt(title, reconstructed_img10, delimiter=',')
    print("Mask=10, MSE=",MSE_10)
    ###########################################################################
    #print("Mask = 20")
    #reconstructed_img20, MSE_20 = test_whole_image(boat,8,20, filter=True, solver="Lasso", display=False, lambda1=0.001)
    #print("Mask=20, MSE=", MSE_20)
    ###########################################################################
    #print("Mask = 30")
    #reconstructed_img30, MSE_30 = test_whole_image(boat,8,30, filter=True, solver="Lasso", display=False, lambda1=0.001)
    #print("Mask=30, MSE=", MSE_30)
    ###########################################################################
    #print("Mask = 40")
    #reconstructed_img40, MSE_40 = test_whole_image(boat,8,40, filter=True, solver="Lasso", display=False, lambda1=0.001)
    #print("Mask=40, MSE=", MSE_40)
    ###########################################################################
    #print("Mask = 50")
    #reconstructed_img50, MSE_50 = test_whole_image(boat,8,50, filter=True, solver="Lasso", display=False, lambda1=0.001)
    #print("Mask=50, MSE=", MSE_50)
    ###########################################################################
    ### Proceed to Graph#######################################################
    #print("Graphing Results")
    #fig, ((ax_1, ax_2, ax_3), (ax_4, ax_5, ax_6)) = plt.subplots(nrows=2, ncols=3, sharex=True)
    # Original Image
    #ax_1.set_title("Original Image")
    #ax_1.imshow(imgRead(boat))
    # Sample = 10
    #ax_2.set_title("Sample = 10")
    #ax_2.imshow(reconstructed_img10)
    # Sample = 20
    #ax_3.set_title("Sample = 20")
    #ax_3.imshow(reconstructed_img20)
    # Sample = 30
    #ax_4.set_title("Sample = 30")
    #ax_4.imshow(reconstructed_img30)
    # Sample = 40
    #ax_5.set_title("Sample = 40")
    #ax_5.imshow(reconstructed_img40)
    # Sample 50
    #ax_6.set_title("Sample = 50")
    #ax_6.imshow(reconstructed_img50)
    #title = "Boat: (Block Size = 8 x 8) & (Median Filtering)"
    #fig.suptitle(title)
    #plt.savefig("Boat8x8_Filtering.png")
    #plt.show()





main_lena_16x16_filtering()
# Test 8x8 with Sample 10
#filename = "Lena_8x8Sample10"
#test_whole_image_KFOLD(boat,16,10,filter=True,solver="Lasso", display=True, lambda1 =0.001,K_FOLD=True, Solve=True, fileName=filename, sampleSize=100)
# Test 8x8 with Sample 20
#filename = "Text8x8Sample20"
#test_whole_image_KFOLD(boat,8,20,filter=True,solver="Lasso", display=True, lambda1 =0.001,K_FOLD=True, Solve=True, fileName=filename)
# Test 8x8 with Sample 30
#filename = "Text8x8Sample30"
#test_whole_image_KFOLD(boat,8,30,filter=True,solver="Lasso", display=True, lambda1 =0.001,K_FOLD=True, Solve=True, fileName=filename)
# Test 8x8 with Sample 40
#filename = "Text8x8Sample40"
#test_whole_image_KFOLD(boat,8,40,filter=True,solver="Lasso", display=True, lambda1 =0.001,K_FOLD=True, Solve=True, fileName=filename)
# Test 8x8 with Sample 50
#filename = "Text8x8Sample50"
#test_whole_image_KFOLD(boat,8,50,filter=True,solver="Lasso", display=True, lambda1 =0.001,K_FOLD=True, Solve=True, fileName=filename)

