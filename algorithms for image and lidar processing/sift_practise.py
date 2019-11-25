# Image Processing
# SIFT(Scale Invariant Feature Transform)
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

# ax, plots = plt.subplots(1, 3, figsize=(5,6))
# plots[0].set_title("0")
# plots[0].imshow(img_RGB)

# adding Scale Invariance and Rotational Invariance
test_img = cv2.pyrDown(img_RGB)
# plots[1].set_title("1")
# plots[1].imshow(test_img)
test_img = cv2.pyrDown(test_img)
# plots[2].set_title("2")
# plots[2].imshow(test_img)
num_rows, num_cols = test_img.shape[:2]

rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 30, 1)
test_img = cv2.warpAffine(test_img, rotation_matrix, (num_cols, num_rows))

test_img_gray = cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY)

# display origin image and testing image
ax, plots = plt.subplots(1, 2, figsize=(6,5))

plots[0].set_title('Training Image')
plots[0].imshow(img_RGB)
plots[1].set_title('Testing Image')
plots[1].imshow(test_img)

sift = cv2.xfeatures2d.SIFT_create()

train_keypoints, train_descriptor = sift.detectAndCompute(img_gray, None)
test_keypoints, test_descriptor = sift.detectAndCompute(test_img_gray, None)

keypoints_without_size = np.copy(img_RGB)
keypoints_with_size = np.copy(img_RGB)

cv2.drawKeypoints(img_RGB, train_keypoints, keypoints_without_size,
                  color=(0,255,0))
cv2.drawKeypoints(img_RGB, train_keypoints, keypoints_with_size,
                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# display image with and without keypoints size
ax, plots = plt.subplots(1, 2, figsize=(10,8))
plots[0].set_title('train keypoints with size')
plots[0].imshow(keypoints_with_size, cmap='gray')
plots[1].set_title('train keypoints without size')
plots[1].imshow(keypoints_without_size)

# number of keypoints detected
print("Number of keypoints detected in the training image: ", len(train_keypoints))
print("Number of keypoints detected in the testing image: ", len(test_keypoints))

# matching keypoints
# create a brute force matcher object
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck = False)

# perfore the matching between the SIFT descriptors of
# the training image and the testimage
matches = bf.match(train_descriptor, test_descriptor)

# the matches with shorter distance are the ones we want
matches = sorted(matches, key = lambda x : x.distance)

result = cv2.drawMatches(img_RGB, train_keypoints,
                         test_img_gray, test_keypoints,
                         matches, test_img_gray, flags=2)

# display the best matching points
plt.rcParams['figure.figsize'] = [14.0, 7.0]
plt.title('Best Matching Points')
plt.imshow(result)
plt.show()
# total number of matching points
print("\nNumber of matching keypoints: ", len(matches))