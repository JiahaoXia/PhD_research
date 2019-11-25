# Image Processing
# BRIEF (Binary Robust Independent Elementary Features
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

star = cv2.xfeatures2d.StarDetector_create()
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
kp1 = star.detect(img_gray, None)
kp2, des = brief.compute(img_gray, kp1)
kp_size1 = np.copy(img_RGB)
kp_size2 = np.copy(img_RGB)
cv2.drawKeypoints(img_RGB, kp1, kp_size1, color=(0,255,0))
cv2.drawKeypoints(img_RGB, kp2, kp_size2, color=(255,255,0))

ax, plots = plt.subplots(1, 2, figsize=(8,6))
plots[0].set_title('star')
plots[0].imshow(kp_size1)

plots[1].set_title('brief')
plots[1].imshow(kp_size2)