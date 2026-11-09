from skimage import data
import matplotlib.pyplot as plt
import numpy as np


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

brik = data.brick()

brik3 = apply_mean_filter(brik, 3)