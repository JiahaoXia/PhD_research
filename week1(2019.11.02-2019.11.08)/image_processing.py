# PhD research week1(2019.11.02-2019.11.08)
# Author: Jiahao Xia
from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os

# Img directory
os.chdir("//media//rise//ESD-USB//Staten Island//Staten Island//20121205A")
test_img = mpimg.imread("survey1_Cam1_000686.jpg")
print(test_img.shape)
plt.figure(figsize=(15, 5))
plt.subplot(1,4,1)
plt.imshow(test_img)
plt.title('RGB')

test_img_R = test_img[:,:,0]
test_img_G = test_img[:,:,1]
test_img_B = test_img[:,:,2]

plt.subplot(1,4,2)
plt.imshow(test_img_R, cmap="gray")
plt.title('R')
plt.subplot(1,4,3)
plt.imshow(test_img_G, cmap="gray")
plt.title('G')
plt.subplot(1,4,4)
plt.imshow(test_img_B, cmap="gray")
plt.title('B')
plt.show()
# save image
plt.savefig("//media//rise//ESD-USB//phd research week1//test_img.png", bbox_inches='tight')


print("***done***")
