import matplotlib.pyplot as plt
import numpy as np

from skimage import io
from skimage.util import random_noise, img_as_float
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

# Image url
apple_image_url = ("https://raw.githubusercontent.com/opencv/opencv/"
"master/samples/data/apple.jpg")
"""
# Load the image URL and apply the salt and pepper filter
appl = io.imread(apple_image_url, as_gray=True)
spappl = random_noise(appl, mode='s&p', amount=0.3)

# filtersize 5
appl5 = apply_mean_filter(appl, 5)
spappl5 = apply_mean_filter(spappl, 5)

#filter size 11
appl11 = apply_mean_filter(appl, 11)
spappl11 = apply_mean_filter(spappl, 11)

# filter size 19
appl19 = apply_mean_filter(appl, 19)
spappl19 = apply_mean_filter(spappl, 19)

# Plot the original image, the salt and pepper image and all the filtered images
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
"""



"""
Task 3, do the same as task 2 but the apple is RGB
"""

"""# it is the same link, so reuse the one from the previous task
appl = np.array(img_as_float(io.imread(apple_image_url))) # Spaghetti, I am yet to find a better and more glorious solution than this
r_appl = appl[:,:,0]
g_appl = appl[:,:,1]
b_appl = appl[:,:,2]

r_appl5 = apply_mean_filter(r_appl, 5)
g_appl5 = apply_mean_filter(g_appl, 5)
b_appl5 = apply_mean_filter(b_appl, 5)
appl5 = np.stack([r_appl5, g_appl5, b_appl5], axis=2)

r_appl11 = apply_mean_filter(r_appl, 11)
g_appl11 = apply_mean_filter(g_appl, 11)
b_appl11 = apply_mean_filter(b_appl, 11)
appl11 = np.stack([r_appl11, g_appl11, b_appl11], axis=2)

r_appl19 = apply_mean_filter(r_appl, 19)
g_appl19 = apply_mean_filter(g_appl, 19)
b_appl19 = apply_mean_filter(b_appl, 19)
appl19 = np.stack([r_appl19, g_appl19, b_appl19], axis=2)


# Apply salt and pepper noise
sandpappl = random_noise(appl, mode='s&p', amount=0.3)
r_sandpaple = sandpappl[:,:,0]
g_sandpaple = sandpappl[:,:,1]
b_sandpaple = sandpappl[:,:,2]

r_sandpaple5 = apply_mean_filter(r_sandpaple, 5)
g_sandpaple5 = apply_mean_filter(g_sandpaple, 5)
b_sandpaple5 = apply_mean_filter(b_sandpaple, 5)
sandpaple5 = np.stack([r_sandpaple5, g_sandpaple5, b_sandpaple5], axis=2)

r_sandpaple11 = apply_mean_filter(r_sandpaple, 11)
g_sandpaple11 = apply_mean_filter(g_sandpaple, 11)
b_sandpaple11 = apply_mean_filter(b_sandpaple, 11)
sandpaple11 = np.stack([r_sandpaple11, g_sandpaple11, b_sandpaple11], axis=2)

r_sandpaple19 = apply_mean_filter(r_sandpaple, 19)
g_sandpaple19 = apply_mean_filter(g_sandpaple, 19)
b_sandpaple19 = apply_mean_filter(b_sandpaple, 19)
sandpaple19 = np.stack([r_sandpaple19, g_sandpaple19, b_sandpaple19], axis=2)

# Plot the original image, the salt and pepper image and all the filtered images
fig, ax = plt.subplots(2, 4)

# Plotting the original image and the filters
ax[0][0].imshow(appl, cmap='gray')
ax[0][1].imshow(appl5, cmap='gray')
ax[0][2].imshow(appl11, cmap='gray')
ax[0][3].imshow(appl19, cmap='gray')

# Plotting the salt and pepper noised image and the filters
ax[1][0].imshow(sandpappl, cmap='gray')
ax[1][1].imshow(sandpaple5, cmap='gray')
ax[1][2].imshow(sandpaple11, cmap='gray')
ax[1][3].imshow(sandpaple19, cmap='gray')

plt.show()
"""

"""
Task 4 Gaussian filter on GRAYSCALE fruits image
- load fruits image¨
- apply a single gaussian filter
- add gaussian noise to the original image
- apply a gaussian filter to denoise the image
- Display results side by side
(Since this is pretty much exactly task 1, I opt to copy paste the function and write a new one, then copy paste task 2)
"""

# The gaussian filter and ites kernel under here -------------------------------------------------------------------------------
def gaussian_kernel(sigma, filter_size):
    if filter_size is None:
        filter_size = 2*int(np.ceil(3*sigma)) + 1

    k = filter_size // 2
    x = np.arange(-k, k + 1)
    xx, yy = np.meshgrid(x, x)
    kernel = np.exp(-(xx*2 + yy*2) / (2 * sigma**2)) # e^((x^2 + y^2) / (2*sigma^2)) - 
    return kernel / kernel.sum() # Normalize so weight sum is 1

def apply_gaussian_filter(image, sigma, filter_size):
    """
    image: 2D array
    size: the size of the mean filter, eg: 3 = 3x3 mean filter
    """
    kernel = gaussian_kernel(sigma, filter_size)
    n = kernel.shape[0]
    k = n // 2

    padded = np.pad(image, k, 'constant', constant_values=0) # pad the image so the negative values don't loop around

    h, w = image.shape
    out = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            wee = padded[i:i+n, j:j+n] # window
            out[i, j] = np.sum(wee * kernel) # weighted average
    return out


fruits_image_url = ("https://raw.githubusercontent.com/opencv/opencv/master/samples/data/fruits.jpg")

frimage = np.array(img_as_float(io.imread(fruits_image_url, as_gray=True)))
gosh_frimage = apply_gaussian_filter(frimage, 2, 3)

goshed_frimage_ohno = random_noise(frimage, mode='gaussian', var=0.01, rng=0)
gosh_goshed_frimage_ohno = apply_gaussian_filter(goshed_frimage_ohno, 2, 3)






fig, ax = plt.subplots(2, 2)

ax[0, 0].imshow(frimage, cmap='gray')
ax[0, 1].imshow(gosh_frimage, cmap='gray')
ax[1, 0].imshow(goshed_frimage_ohno, cmap='gray')
ax[1, 1].imshow(gosh_goshed_frimage_ohno, cmap='gray')
plt.show()

