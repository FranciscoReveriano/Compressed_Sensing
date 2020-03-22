import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from util.util import *

# Load Images
Boat_Mask_10 = np.loadtxt("/home/franciscoAML/Documents/Compressed_Sensing/Boat_No_FIltering/Boat_Mask_10.txt", delimiter=",")
Boat_Mask_20 = np.loadtxt("/home/franciscoAML/Documents/Compressed_Sensing/Boat_No_FIltering/Boat_Mask_20.txt", delimiter=",")
Boat_Mask_30 = np.loadtxt("/home/franciscoAML/Documents/Compressed_Sensing/Boat_No_FIltering/Boat_Mask_30.txt", delimiter=",")
Boat_Mask_40 = np.loadtxt("/home/franciscoAML/Documents/Compressed_Sensing/Boat_No_FIltering/Boat_Mask_40.txt", delimiter=",")
Boat_Mask_50 = np.loadtxt("/home/franciscoAML/Documents/Compressed_Sensing/Boat_No_FIltering/Boat_Mask_50.txt", delimiter=",")

# Determine Values For Median Filter
dimension = (8,8)
original_image = plt.imread("/home/franciscoAML/Documents/Compressed_Sensing/fishing_boat.bmp")
P,Q = original_image.shape[0], original_image.shape[1]
filter_Boat_Mask_10 = medianFilter(Boat_Mask_10, dimension, P,Q)
filter_Boat_Mask_20 = medianFilter(Boat_Mask_20, dimension, P,Q)
filter_Boat_Mask_30 = medianFilter(Boat_Mask_30, dimension, P,Q)
filter_Boat_Mask_40 = medianFilter(Boat_Mask_40, dimension, P,Q)
filter_Boat_Mask_50 = medianFilter(Boat_Mask_50, dimension, P,Q)

def plot_non_filtered():
    ### Proceed to Graph#######################################################
    print("Graphing Results")
    fig, ((ax_1, ax_2, ax_3), (ax_4, ax_5, ax_6)) = plt.subplots(nrows=2, ncols=3, sharex=True)
    # Original Image
    ax_1.set_title("Original Image")
    ax_1.imshow(plt.imread("/home/franciscoAML/Documents/Compressed_Sensing/fishing_boat.bmp"))
    # Sample = 10
    ax_2.set_title("Sample = 10")
    ax_2.imshow(Boat_Mask_10)
    # Sample = 20
    ax_3.set_title("Sample = 20")
    ax_3.imshow(Boat_Mask_20)
    # Sample = 30
    ax_4.set_title("Sample = 30")
    ax_4.imshow(Boat_Mask_30)
    # Sample = 40
    ax_5.set_title("Sample = 40")
    ax_5.imshow(Boat_Mask_40)
    # Sample 50
    ax_6.set_title("Sample = 50")
    ax_6.imshow(Boat_Mask_50)
    title = "Boat: (Block Size = 8 x 8) & (Median Filter) & (Optimal Lambda)"
    fig.suptitle(title)
    #plt.savefig("/home/franciscoAML/Documents/Compressed_Sensing/Lena_16x16_No_Filtering_2/Lena_16x16_No_Filtering.png")
    plt.show()

    # Calculate MSE
    original_image = plt.imread("/home/franciscoAML/Documents/Compressed_Sensing/fishing_boat.bmp")
    ### Calculate MSE
    print("Sample 10:", mean_squared_error(original_image, Boat_Mask_10))
    print("Sample 20:", mean_squared_error(original_image, Boat_Mask_20))
    print("Sample 30:", mean_squared_error(original_image, Boat_Mask_30))
    print("Sample 40:", mean_squared_error(original_image, Boat_Mask_40))
    print("Sample 50:", mean_squared_error(original_image, Boat_Mask_50))


def plot_filtered():
    ### Proceed to Graph#######################################################
    print("Graphing Results")
    fig, ((ax_1, ax_2, ax_3), (ax_4, ax_5, ax_6)) = plt.subplots(nrows=2, ncols=3, sharex=True)
    # Original Image
    ax_1.set_title("Original Image")
    ax_1.imshow(plt.imread("/home/franciscoAML/Documents/Compressed_Sensing/fishing_boat.bmp"))
    # Sample = 10
    ax_2.set_title("Sample = 10")
    ax_2.imshow(Boat_Mask_10)
    # Sample = 20
    ax_3.set_title("Sample = 20")
    ax_3.imshow(Boat_Mask_20)
    # Sample = 30
    ax_4.set_title("Sample = 30")
    ax_4.imshow(Boat_Mask_30)
    # Sample = 40
    ax_5.set_title("Sample = 40")
    ax_5.imshow(Boat_Mask_40)
    # Sample 50
    ax_6.set_title("Sample = 50")
    ax_6.imshow(Boat_Mask_50)
    title = "Boat: (Block Size = 8 x 8) & (Median Filter) & (Optimal Lambda)"
    fig.suptitle(title)
    # plt.savefig("/home/franciscoAML/Documents/Compressed_Sensing/Lena_16x16_No_Filtering_2/Lena_16x16_No_Filtering.png")
    plt.show()

    # Calculate MSE
    original_image = plt.imread("/home/franciscoAML/Documents/Compressed_Sensing/fishing_boat.bmp")
    ### Calculate MSE
    print("Sample 10:", mean_squared_error(original_image, filter_Boat_Mask_10))
    print("Sample 20:", mean_squared_error(original_image, filter_Boat_Mask_20))
    print("Sample 30:", mean_squared_error(original_image, filter_Boat_Mask_30))
    print("Sample 40:", mean_squared_error(original_image, filter_Boat_Mask_40))
    print("Sample 50:", mean_squared_error(original_image, filter_Boat_Mask_50))


plot_filtered()