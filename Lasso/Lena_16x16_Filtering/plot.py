import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from util.util import *

# Load Images
Lena_Mask_10 = np.loadtxt("/home/franciscoAML/Documents/Compressed_Sensing/Lena_16x16_Filtering/Lena_Mask_10.txt", delimiter=",")
Lena_Mask_30 = np.loadtxt("/home/franciscoAML/Documents/Compressed_Sensing/Lena_16x16_Filtering/Lena_Mask_30.txt", delimiter=",")
Lena_Mask_50 = np.loadtxt("/home/franciscoAML/Documents/Compressed_Sensing/Lena_16x16_Filtering/Lena_Mask_50.txt", delimiter=",")
Lena_Mask_100 = np.loadtxt("/home/franciscoAML/Documents/Compressed_Sensing/Lena_16x16_Filtering/Lena_Mask_100.txt", delimiter=",")
Lena_Mask_150 = np.loadtxt("/home/franciscoAML/Documents/Compressed_Sensing/Lena_16x16_Filtering/Lena_Mask_150.txt", delimiter=",")

# Determine Values For Median Filter
dimension = (16,16)
original_image = plt.imread("/home/franciscoAML/Documents/Compressed_Sensing/lena.bmp")[:,:,0]
P,Q = original_image.shape[0], original_image.shape[1]
filter_Lena_Mask_10 = medianFilter(Lena_Mask_10, dimension, P,Q)
filter_Lena_Mask_30 = medianFilter(Lena_Mask_30, dimension, P,Q)
filter_Lena_Mask_50 = medianFilter(Lena_Mask_50, dimension, P,Q)
filter_Lena_Mask_100 = medianFilter(Lena_Mask_100, dimension, P,Q)
filter_Lena_Mask_150 = medianFilter(Lena_Mask_150, dimension, P,Q)


def plot_no_filter():
    ### Proceed to Graph#######################################################
    print("Graphing Results")
    fig, ((ax_1, ax_2, ax_3), (ax_4, ax_5, ax_6)) = plt.subplots(nrows=2, ncols=3, sharex=True)
    # Original Image
    ax_1.set_title("Original Image")
    ax_1.imshow(plt.imread("/home/franciscoAML/Documents/Compressed_Sensing/lena.bmp")[:,:,0])
    # Sample = 10
    ax_2.set_title("Sample = 10")
    ax_2.imshow(Lena_Mask_10)
    # Sample = 20
    ax_3.set_title("Sample = 30")
    ax_3.imshow(Lena_Mask_30)
    # Sample = 30
    ax_4.set_title("Sample = 50")
    ax_4.imshow(Lena_Mask_50)
    # Sample = 40
    ax_5.set_title("Sample = 100")
    ax_5.imshow(Lena_Mask_100)
    # Sample 50
    ax_6.set_title("Sample = 150")
    ax_6.imshow(Lena_Mask_150)
    title = "Lena: (Block Size = 16 x 16) & (No Filtering) & (Optimal Lambda)"
    fig.suptitle(title)
    plt.savefig("/home/franciscoAML/Documents/Compressed_Sensing/Lena_16x16_Filtering/Lena_16x16_No_Filtering.png")


    # Calculate MSE
    original_image = plt.imread("/home/franciscoAML/Documents/Compressed_Sensing/lena.bmp")[:,:,0]
    ### Calculate MSE
    print("Sample 10:", mean_squared_error(original_image,Lena_Mask_10))
    print("Sample 30:", mean_squared_error(original_image, Lena_Mask_30))
    print("Sample 50:", mean_squared_error(original_image, Lena_Mask_50))
    print("Sample 100:", mean_squared_error(original_image, Lena_Mask_100))
    print("Sample 150:", mean_squared_error(original_image, Lena_Mask_150))


def plot_filter():
    ### Proceed to Graph#######################################################
    print("Graphing Results")
    fig, ((ax_1, ax_2, ax_3), (ax_4, ax_5, ax_6)) = plt.subplots(nrows=2, ncols=3, sharex=True)
    # Original Image
    ax_1.set_title("Original Image")
    ax_1.imshow(plt.imread("/home/franciscoAML/Documents/Compressed_Sensing/lena.bmp")[:,:,0])
    # Sample = 10
    ax_2.set_title("Sample = 10")
    ax_2.imshow(filter_Lena_Mask_10)
    # Sample = 20
    ax_3.set_title("Sample = 30")
    ax_3.imshow(filter_Lena_Mask_30)
    # Sample = 30
    ax_4.set_title("Sample = 50")
    ax_4.imshow(filter_Lena_Mask_50)
    # Sample = 40
    ax_5.set_title("Sample = 100")
    ax_5.imshow(filter_Lena_Mask_100)
    # Sample 50
    ax_6.set_title("Sample = 150")
    ax_6.imshow(filter_Lena_Mask_150)
    title = "Lena: (Block Size = 16 x 16) & (Filtering) & (Optimal Lambda)"
    fig.suptitle(title)
    plt.savefig("/home/franciscoAML/Documents/Compressed_Sensing/Lena_16x16_Filtering/Lena_16x16_Filtering.png")


    # Calculate MSE
    original_image = plt.imread("/home/franciscoAML/Documents/Compressed_Sensing/lena.bmp")[:,:,0]
    ### Calculate MSE
    print("Sample 10:", mean_squared_error(original_image, filter_Lena_Mask_10))
    print("Sample 20:", mean_squared_error(original_image, filter_Lena_Mask_30))
    print("Sample 30:", mean_squared_error(original_image, filter_Lena_Mask_50))
    print("Sample 40:", mean_squared_error(original_image, filter_Lena_Mask_100))
    print("Sample 50:", mean_squared_error(original_image, filter_Lena_Mask_150))


plot_filter()