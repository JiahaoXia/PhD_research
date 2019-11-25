# image processing
# ORB (Oriented FAST and Rotated BRIEF)
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir("F://Staten Island//Staten Island")
# read image
img = cv2.imread("20121205A//survey1_Cam1_000230.jpg")
# --->RGB
img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# --->gray
img_gray = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2GRAY)

# Initiate Star detector
orb = cv2.ORB_create()
# find keypoints with ORB
kp = orb.detect(img_gray, None)
# compute the descriptors with ORB
kp, des = orb.compute(img_gray, kp)
# draw only keypoints location, not size and orientation
img2 = np.copy(img_RGB)
cv2.drawKeypoints(img_RGB, kp, img2, color=(0,255,0), flags=0)
plt.imshow(img2)
plt.show()