import matplotlib.pyplot as plt
import numpy as np

from skimage import io
from skimage.util import random_noise
from skimage import data


"""
Task 1:
Implement mean filtering and ivestigate how filter size affects image smoothing

- Load grayscale brick image
- Create a reusable function apply_mean_filter
- Apply the function to the image using these sizes: 3x3 9x9 15x15
- Display the original image next to the smoothed images to compare
"""

# The mean filter function
def apply_mean_filter(image, filter_size):
    """
    image: 2D array
    size: the size of the mean filter, eg: 3 = 3x3 mean filter
    """
    k = filter_size // 2 # given that the size input is an odd number, this will work
    padded = np.pad(image, k, 'constant', constant_values=0) # pad the image so the negative values don't loop around

    h, w = image.shape
    out = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            out[i][j] = np.sum(padded[i+k-k:i+k+k+1, j+k-k:j+k+k+1]) / (filter_size**2) # Spaghetti fixes everything

    return out

"""
brik = data.brick()

# Apply the mean average filter
brik3 = apply_mean_filter(brik, 3)
brik9 = apply_mean_filter(brik, 9)
brik15 = apply_mean_filter(brik, 15)

fig, ax = plt.subplots(1, 4)

ax[0].imshow(brik, cmap='gray')
ax[1].imshow(brik3, cmap='gray')
ax[2].imshow(brik9, cmap='gray')
ax[3].imshow(brik15, cmap='gray')

plt.show()
"""


"""
The end of task one --------------------------------------------------------------------------------
"""


"""
Task 2
Apply the mean_average_filter to the apple image
implement the mean filtering and 'investigate' how the filter size affects smoothing.
- load apple image
- apply 3 different sizes of mean filtering
- add salt and pepper noise to the original image
- display all results
"""

image_url = ("https://raw.githubusercontent.com/opencv/opencv/"
"master/samples/data/apple.jpg")

appl = io.imread(image_url, as_gray=True)
spappl = random_noise(appl, mode='s&p', amount=0.1)

# filtersize 5
appl5 = apply_mean_filter(appl, 5)
spappl5 = apply_mean_filter(spappl, 5)

#filter size 11
appl11 = apply_mean_filter(appl, 11)
spappl11 = apply_mean_filter(spappl, 11)

# filter size 19
appl19 = apply_mean_filter(appl, 19)
spappl19 = apply_mean_filter(spappl, 19)


fig, ax = plt.subplots(2, 4)

# Plotting the original image and the filters
ax[0][0].imshow(appl, cmap='gray')
ax[0][1].imshow(appl5, cmap='gray')
ax[0][2].imshow(appl11, cmap='gray')
ax[0][3].imshow(appl19, cmap='gray')

# Plotting the salt and pepper noised image and the filters
ax[1][0].imshow(spappl, cmap='gray')
ax[1][1].imshow(spappl5, cmap='gray')
ax[1][2].imshow(spappl11, cmap='gray')
ax[1][3].imshow(spappl19, cmap='gray')

plt.show()

