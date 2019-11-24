# Image Processing
# Fast (Features from Accelerated Segment Test)
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

ax, plots = plt.subplots(1, 2, figsize=(10,8))

plots[0].set_title("Origin Image")
plots[0].imshow(img_RGB)

plots[1].set_title("Gray Image")
plots[1].imshow(img_gray, cmap="gray")

fast = cv2.FastFeatureDetector_create()

# Detect keypoints with non max suppression
keypoints_with_nonmax = fast.detect(img_gray, None)

# Disable nonmaxSuppression
fast.setNonmaxSuppression(False)
# Detect keypoints without non max suppression
keypoints_without_nonmax = fast.detect(img_gray, None)

image_with_nonmax = np.copy(img_RGB)
image_without_nonmax = np.copy(img_RGB)

# Draw keypoints on top of the input image
cv2.drawKeypoints(img_RGB, keypoints_with_nonmax, image_with_nonmax,
                  color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.drawKeypoints(img_RGB, keypoints_without_nonmax, image_without_nonmax,
                  color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Display image with keypoints
ax, plots = plt.subplots(1, 2, figsize=(20,10))

plots[0].set_title("with non max suppression")
plots[0].imshow(image_with_nonmax)

plots[1].set_title("without non max suppression")
plots[1].imshow(image_without_nonmax)

# print the number of keypoints
print("Number of Keypoints(with non max suppression): ", len(keypoints_with_nonmax))
print("Number of Keypoints(without non max suppression): ", len(keypoints_without_nonmax))