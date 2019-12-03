# Image Lens Distortion
# Author: Jiahao Xia
import os
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt

def lens_distortion_correction(image, A, B, C):
    image_b, image_g, image_r = cv2.split(image)
    height = image_b.shape[0]
    width = image_b.shape[1]
    image_x = np.arange(height).reshape(-1,1).repeat(width, axis=1)
    image_rx = np.concatenate((np.arange(-1*height/2,0,1), np.arange(1,height/2+1,1)), axis=0)
    image_rx = image_rx.reshape(-1,1).repeat(width, axis=1)
    image_y = np.arange(width).reshape(1,-1).repeat(height, axis=0)
    image_ry = np.concatenate((np.arange(-1*width/2,0,1), np.arange(1,width/2+1,1)), axis=0)
    image_ry = image_ry.reshape(1,-1).repeat(height, axis=0)
    image_d2 = image_rx*image_rx + image_ry*image_ry
    image_d4 = image_d2*image_d2
    image_d6 = image_d2*image_d4
    image_dx = -1*image_rx*((A*image_d2) + (B*image_d4) + (C*image_d6))
    image_dy = -1*image_ry*((A*image_d2) + (B*image_d4) + (C*image_d6))

    image_x_correct = image_x + image_dx
    image_y_correct = image_y + image_dy

    image_x_correct_int = np.trunc(image_x_correct)
    image_x_correct_int = image_x_correct_int.astype(int)
    image_x_correct_int[(image_x_correct_int<0) | (image_x_correct_int>height-1)] = image_x[(image_x_correct_int<0) | (image_x_correct_int>height-1)]
    image_y_correct_int = np.trunc(image_y_correct)
    image_y_correct_int = image_y_correct_int.astype(int)
    image_y_correct_int[(image_y_correct_int<0) | (image_y_correct_int>width-1)] = image_y[(image_y_correct_int<0) | (image_y_correct_int>width-1)]

    image_correction = image
    image_correction[image_x_correct_int.reshape(1,-1), image_y_correct_int.reshape(1,-1), 0] = image_b[image_x.reshape(1,-1), image_y.reshape(1,-1)]
    image_correction[image_x_correct_int.reshape(1,-1), image_y_correct_int.reshape(1,-1), 1] = image_g[image_x.reshape(1,-1), image_y.reshape(1,-1)]
    image_correction[image_x_correct_int.reshape(1,-1), image_y_correct_int.reshape(1,-1), 2] = image_r[image_x.reshape(1,-1), image_y.reshape(1,-1)]

    return image_correction

os.chdir('F://Staten Island//Staten Island//')
# cameras information files
cam_info = ['camera1_cal.csv', 'camera2_cal.csv', 'camera3_cal.csv', 'camera4_cal.csv']
# Image LAS match
img2block = 'img2block.csv'
img2block_data = pd.read_csv(img2block)
img_num = img2block_data.shape[0]

for i in range(1):
    temp_image_file = img2block_data['ImageName'][i]
    temp_image = cv2.imread('20121205A//' + temp_image_file)

    temp_cam_no = int(temp_image_file.split('_')[1][-1])
    temp_cam_file = cam_info[temp_cam_no - 1]
    temp_cam_info = pd.read_csv(temp_cam_file)
    temp_A = temp_cam_info['A3'].values
    temp_B = temp_cam_info['A5'].values
    temp_C = temp_cam_info['A7'].values

    temp_image_correction = lens_distortion_correction(temp_image, temp_A, temp_B, temp_C)
    temp_image_correction = cv2.cvtColor(temp_image_correction, cv2.COLOR_BGR2RGB)
    plt.imshow(temp_image_correction)
    plt.show()