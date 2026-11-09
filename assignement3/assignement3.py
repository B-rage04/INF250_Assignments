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
def apply_mean_filter(image, size):
    """
    image: 2D array
    size: the size of the mean filter, eg: 3 = 3x3 mean filter
    """
    for y in image:
        for x in y:
            print(x)

brik = data.brick()

brik3 = apply_mean_filter(brik, 3)