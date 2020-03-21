import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Load Images
Lena_Mask_10 = np.loadtxt("/home/franciscoAML/Documents/Compressed_Sensing/Lena_16x16_No_Filtering_2/Lena_Mask_10.txt", delimiter=",")
Lena_Mask_20 = np.loadtxt("/home/franciscoAML/Documents/Compressed_Sensing/Lena_16x16_No_Filtering_2/Lena_Mask_20.txt", delimiter=",")
Lena_Mask_30 = np.loadtxt("/home/franciscoAML/Documents/Compressed_Sensing/Lena_16x16_No_Filtering_2/Lena_Mask_30.txt", delimiter=",")
Lena_Mask_40 = np.loadtxt("/home/franciscoAML/Documents/Compressed_Sensing/Lena_16x16_No_Filtering_2/Lena_Mask_40.txt", delimiter=",")
Lena_Mask_50 = np.loadtxt("/home/franciscoAML/Documents/Compressed_Sensing/Lena_16x16_No_Filtering_2/Lena_Mask_50.txt", delimiter=",")

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
ax_3.set_title("Sample = 20")
ax_3.imshow(Lena_Mask_20)
# Sample = 30
ax_4.set_title("Sample = 30")
ax_4.imshow(Lena_Mask_30)
# Sample = 40
ax_5.set_title("Sample = 40")
ax_5.imshow(Lena_Mask_40)
# Sample 50
ax_6.set_title("Sample = 50")
ax_6.imshow(Lena_Mask_50)
title = "Lena: (Block Size = 16 x 16) & (No Filtering) & (Optimal Lambda)"
fig.suptitle(title)
plt.savefig("/home/franciscoAML/Documents/Compressed_Sensing/Lena_16x16_No_Filtering_2/Lena_16x16_No_Filtering.png")
plt.show()

# Calculate MSE
original_image = plt.imread("/home/franciscoAML/Documents/Compressed_Sensing/lena.bmp")[:,:,0]
### Calculate MSE
print("Sample 10:", mean_squared_error(original_image,Lena_Mask_10))
print("Sample 20:", mean_squared_error(original_image, Lena_Mask_20))
print("Sample 30:", mean_squared_error(original_image, Lena_Mask_30))
print("Sample 40:", mean_squared_error(original_image, Lena_Mask_40))
print("Sample 50:", mean_squared_error(original_image, Lena_Mask_50))