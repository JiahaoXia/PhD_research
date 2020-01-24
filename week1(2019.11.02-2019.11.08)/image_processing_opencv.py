# PhD research week1(2019.11.02-2019.11.08)
# Author: Jiahao Xia
from __future__ import print_function
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Img directory
os.chdir("//media//rise//ESD-USB//Staten Island//Staten Island//20121205A")
# read single channel image
test_img = cv2.imread("survey1_Cam1_000686.jpg", 0)
hist = cv2.calcHist([test_img], [0], None, [256], [0.0,255.0])
x1 = np.arange(0,256,1)
plt.plot(x1, hist)
plt.axis([0,255,0,160000])
plt.show()
# cv2.imshow('hist', histImg)
# cv2.waitKey(0)

# read RGB image
test_img = cv2.imread("survey1_Cam1_000686.jpg")
b, g, r = cv2.split(test_img)
