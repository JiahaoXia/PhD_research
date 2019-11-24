# Image Processing
# Harris
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

os.chdir("F://Staten Island//Staten Island")
# read image
img = cv2.imread("20121205A//survey1_Cam1_000230.jpg")
# --->RGB
img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# --->gray
img_gray = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2GRAY)

# find harris corners
img_gray = np.float32(img_gray)
dst = cv2.cornerHarris(img_gray, 2, 3, 0.04)
dst = cv2.dilate(dst, None)
ret, dst = cv2.threshold(dst, 0.01*dst.max(), 255, 0)
# img[dst>0.01*dst.max()] = [0,255,255]
# cv2.imshow('dst', img)
dst = np.uint8(dst)

# find centroids
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

# create the criteria to stop and refine the corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv2.cornerSubPix(img_gray, np.float32(centroids), (5,5), (-1,-1), criteria)

# Now draw them
res = np.hstack((centroids, corners))
res = np.int0(res)
img[res[:, 1], res[:, 0]] = [0,0,255]
img[res[:, 3], res[:, 2]] = [0,255,2]

cv2.imshow('img', img)