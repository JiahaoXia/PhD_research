# Image Processing
# SURF(Speeded-Up Robust Features)
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

os.chdir("F://Staten Island//Staten Island")
# read image
img = cv2.imread("20121205A//survey1_Cam1_000230.jpg")
# --->RGB
img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# --->gray
img_gray = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2GRAY)

surf = cv2.xfeatures2d.SURF_create()

keypoints, descriptor = surf.detectAndCompute(img_gray, None)

keypoints_size = np.copy(img_RGB)

cv2.drawKeypoints(img_RGB, keypoints, keypoints_size, color=(0,255,0))

plt.imshow(keypoints_size)