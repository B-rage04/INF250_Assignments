# By Brage Bestvold and Lasse Lindholm

from skimage import data, exposure
import matplotlib.pyplot as plt
import numpy as np


"""
## Question 1

blob = data.binary_blobs()

blob = blob.astype(np.uint8) * 255

num_white = int(np.sum(blob) / 255)
num_black = int(blob.size - num_white)

fig, ax = plt.subplots(1, 2, figsize=(10,5))

ax[0].bar(["Black Pixels", "White Pixels"], [num_black, num_white])
ax[0].set_title("Histogram Blob image")
ax[1].imshow(blob, vmin=0, cmap="gray", vmax=255)
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_title("Blob image")
plt.show()

percent_white = num_white / blob.size * 100
percent_black = num_black / blob.size * 100
print("Question 1")
print(f"{percent_white}% of the image is white")
print(f"{percent_black}% of the image is black")
print("*"*30)



## Question 2

coins = data.coins()
print("Question 2")
print(f"Datatype: {coins.dtype}")

print(f"Min value: {np.min(coins)}")
print(f"Max value: {np.max(coins)}")
print("All pixels lie in the range from 0 to 255")

fig, ax = plt.subplots(1,2, figsize=(10,5))
ax[0].hist(coins.ravel(), bins=256, range=(0,255))
ax[0].set_title("Coins histogram")


ax[1].imshow(coins, vmin=0, cmap="gray", vmax=255)
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_title("Coins image")
print(f"Mean pixel value: {np.mean(coins)}")
print(f"Mode: {np.argmax(np.bincount(coins.ravel()))}")
print("*"*30)
plt.show()



## Question 3

brick = data.brick()

fig, ax = plt.subplots(1,2, figsize=(10,5))
ax[0].hist(brick.ravel(), bins=256, range=(0,255), density=True)
ax[0].set_title("Brick histogram")


ax[1].imshow(brick, vmin=0, cmap="gray", vmax=255)
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_title("Brick image")
plt.show()


## Question 4

chelsea = data.chelsea()

r = chelsea[:,:,0]
g = chelsea[:,:,1]
b = chelsea[:,:,2]

r = r.astype(np.float64)
g = g.astype(np.float64)
b = b.astype(np.float64)

w_R = 0.2126
w_G = 0.7152
w_B = 0.0722


chelsea_gray = w_R * r + w_G * g + w_B * b

chelsea_gray = np.clip(chelsea_gray, 0, 255)
chelsea_gray = chelsea_gray.astype(np.uint8)

fig, ax = plt.subplots(1,2, figsize=(10,5))
ax[0].imshow(chelsea_gray, vmin=0, cmap="gray", vmax=255)
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_title("Chelsea grayscale")

ax[1].imshow(chelsea)
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_title("Chelsea color")
plt.show()

## Question 5

fig, ax = plt.subplots(2,3, figsize = (15,10))
ax[0,0].imshow(r, vmin=0, cmap="gray", vmax=255)
ax[0,0].set_title("Redscale")
ax[0,0].set_xticks([])
ax[0,0].set_yticks([])
ax[0,1].imshow(g, vmin=0, cmap="gray", vmax=255)
ax[0,1].set_title("Greenscale")
ax[0,1].set_xticks([])
ax[0,1].set_yticks([])
ax[0,2].imshow(b, vmin=0, cmap="gray", vmax=255)
ax[0,2].set_title("Bluescale")
ax[0,2].set_xticks([])
ax[0,2].set_yticks([])

ax[1,0].hist(r.ravel(), bins=256, range=(0,255), color="Red")
ax[1,0].set_title("Redscale histogram")
ax[1,1].hist(g.ravel(), bins=256, range=(0,255), color="Green")
ax[1,1].set_title("Greenscale histogram")
ax[1,2].hist(b.ravel(), bins=256, range=(0,255), color="Blue")
ax[1,2].set_title("Bluescale histogram")


plt.show()

## Question 6

coins = data.coins()
coins_equ = exposure.equalize_hist(coins)

coins_equ = coins_equ * 255
coins_equ = np.clip(coins_equ, 0, 255)
coins_equ = coins_equ.astype(np.uint8)

fig, ax = plt.subplots(2,2, figsize = (10,10))

ax[0,0].hist(coins.ravel(), bins=256, range=(0,255))
ax[0,0].set_title("coins histogram")


ax[0,1].imshow(coins, vmin=0, cmap="gray", vmax=255)
ax[0,1].set_xticks([])
ax[0,1].set_yticks([])
ax[0,1].set_title("Coins image")

ax[1,0].hist(coins_equ.ravel(), bins=256, range=(0,255))
ax[1,0].set_title("Equalized coins histogram")


ax[1,1].imshow(coins_equ, vmin=0, cmap="gray", vmax=255)
ax[1,1].set_xticks([])
ax[1,1].set_yticks([])
ax[1,1].set_title("Equalized coins image")

plt.show()


## Question 7

micro = data.microaneurysms()

micro_equ = exposure.equalize_hist(micro)

micro_equ = micro_equ * 255
micro_equ = micro_equ.astype(np.uint8)

fig, ax = plt.subplots(2,2, figsize = (10,10))

ax[0,0].hist(micro.ravel(), bins=256, range=(0,255))
ax[0,0].set_title("Microaneurysms histogram")


ax[0,1].imshow(micro, vmin=0, cmap="gray", vmax=255)
ax[0,1].set_xticks([])
ax[0,1].set_yticks([])
ax[0,1].set_title("Microaneurysms image")

ax[1,0].hist(micro_equ.ravel(), bins=256, range=(0,255))
ax[1,0].set_title("Equalized microaneurysms histogram")


ax[1,1].imshow(micro_equ, vmin=0, cmap="gray", vmax=255)
ax[1,1].set_xticks([])
ax[1,1].set_yticks([])
ax[1,1].set_title("Equalized microaneurysms image")

plt.show()
"""
## Question 8

gravel = data.gravel()

gravel_equ = exposure.equalize_hist(gravel)

gravel_equ = gravel_equ * 255
gravel_equ = gravel_equ.astype(np.uint8)

fig, ax = plt.subplots(2,2, figsize = (10,10))

ax[0,0].hist(gravel.ravel(), bins=256, range=(0,255))
ax[0,0].set_title("Gravel histogram")


ax[0,1].imshow(gravel, vmin=0, cmap="gray", vmax=255)
ax[0,1].set_xticks([])
ax[0,1].set_yticks([])
ax[0,1].set_title("Gravel image")

ax[1,0].hist(gravel_equ.ravel(), bins=256, range=(0,255))
ax[1,0].set_title("Equalized gravel histogram")


ax[1,1].imshow(gravel_equ, vmin=0, cmap="gray", vmax=255)
ax[1,1].set_xticks([])
ax[1,1].set_yticks([])
ax[1,1].set_title("Equalized gravel image")

plt.show()
"""
"""