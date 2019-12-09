# The Transformation of Translation and Rotation
# From Image to Point Cloud
# Author: Jiahao Xia
import os
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from laspy.file import File
os.chdir('F://Staten Island//Staten Island//')

def camera_info_read(cam_file):
    """
    get the heading, roll, pitch, x0, y0, f
    :param cam_file:
    :return: pd.Dataframe
    """
    cam_info = pd.read_csv(cam_file)
    cam_heading = cam_info['AttitudeC_heading'].values[0]
    cam_roll = cam_info['AttitudeC_roll'].values[0]
    cam_pitch = cam_info['AttitudeC_pitch'].values[0]
    cam_X = cam_info['PrincipalP_X'].values[0] * 0.000006
    cam_Y = cam_info['PrincipalP_Y'].values[0] * 0.000006
    cam_Z = cam_info['PrincipalP_Z'].values[0] * 0.000006
    result = pd.DataFrame({'heading': [cam_heading], 'roll': [cam_roll], 'pitch': [cam_pitch],
                           'x0': [cam_X], 'y0': [cam_Y], 'f': [cam_Z]})
    return result

def cal_rotation_matrix(heading, roll, pitch):
    """
    Calculate the rotation matrix according to the heading, roll and pitch
    :param heading:
    :param roll:
    :param pitch:
    :return:
    """
    a11 = math.cos(heading)*math.cos(pitch)-math.sin(heading)*math.sin(roll)*math.sin(pitch)
    a12 = -1*math.cos(heading)*math.sin(pitch)-math.sin(heading)*math.sin(roll)*math.cos(pitch)
    a13 = -1*math.sin(heading)*math.cos(roll)
    a21 = math.cos(roll)*math.sin(pitch)
    a22 = math.cos(roll)*math.cos(pitch)
    a23 = -1*math.sin(roll)
    a31 = math.sin(heading)*math.cos(pitch)+math.cos(heading)*math.sin(roll)*math.sin(pitch)
    a32 = -1*math.sin(heading)*math.sin(pitch)+math.cos(heading)*math.sin(roll)*math.cos(pitch)
    a33 = math.cos(heading)*math.cos(roll)
    result = np.array([[a11,a12,a13],[a21,a22,a23],[a31,a32,a33]])
    return result

def cal_rotation_matrix_xyz(angle_x, angle_y, angle_z):
    """
    calculate the rotation matrix according to the rotation angle of x,y,z axis
    :param angle_x: the rotation angle of x axis
    :param angle_y: the rotation angle of y axis
    :param angle_z: the rotation angle of z axis
    :return: return the rotation matrix
    """
    rotation_matrix_x = np.array([[1, 0, 0],
                                  [0, math.cos(angle_x), -1*math.sin(angle_x)],
                                  [0, math.sin(angle_x), math.cos(angle_x)]])
    rotation_matrix_y = np.array([[math.cos(angle_y), 0, -1*math.sin(angle_y)],
                                  [0, 1, 0],
                                  [math.sin(angle_y), 0, math.cos(angle_y)]])
    rotation_matrix_z = np.array([[math.cos(angle_z), -1*math.sin(angle_z), 0],
                                  [math.sin(angle_z), math.cos(angle_z), 0],
                                  [0, 0, 1]])
    rotation_matrix = np.dot(rotation_matrix_x, rotation_matrix_y, rotation_matrix_z)
    return rotation_matrix

def img_read_processing(img_file, x0, y0, f, pixel_size):
    """
    reading and processing image
    :param img_file: file of image
    :param x0: the x position of the principal point relative to the center of image
    :param y0: the y position of the principal point relative to the center of image
    :param f: the focus length
    :param pixel_size: the size of the pixel
    :return:
    """
    img = cv2.imread(img_file)
    height = img.shape[0]
    width = img.shape[1]

    img_x = np.concatenate((np.arange(-0.5-height/2, -0.5, 1), np.arange(0.5, 0.5+height/2, 1)), axis=0) * pixel_size
    img_x = img_x.reshape(-1,1).repeat(width, axis=1)
    img_x = img_x - x0

    img_y = np.concatenate((np.arange(-0.5-width/2, -0.5, 1), np.arange(0.5, 0.5+width/2, 1)), axis=0) * pixel_size
    img_y = img_y.reshape(-1,1).repeat(height, axis=0)
    img_y = img_y - y0

    # image pixel coordinate (x, y, -f) 3*n(number of pixel)
    img_matrix = np.concatenate((img_x.reshape(1,-1), img_y.reshape(1,-1), np.full((1,height*width), -1*f)), axis=0)

    return img_matrix

def img_xy_length2pixel(x, y, pixel_size, heigth, width):
    """
    convert the image coordinate from length(meter) to pixel
    :param x: numpy array, the length coordinate
    :param y: numpy array, the length coordinate
    :param piexl_size: the size of pixel
    :param height: the height of image
    :param width: the width of image
    :return: the pixel coordinate (the origin should be the left-up corner)
    """
    out_x = np.round((x + 0.5 * height * pixel_size) / pixel_size)
    out_y = np.round((y + 0.5 * width * pixel_size) / pixel_size)

    out_x = out_x.astype(int)
    out_y = out_y.astype(int)

    return out_x, out_y

if __name__ == '__main__':
    img_info_file = 'camera1_cal.csv'
    cam_info_data = camera_info_read(img_info_file)
    cam_heading = cam_info_data['heading'].values[0]
    cam_roll = cam_info_data['roll'].values[0]
    cam_pitch = cam_info_data['pitch'].values[0]
    cam_x0 = cam_info_data['x0'].values[0]
    cam_y0 = cam_info_data['y0'].values[0]
    cam_f = cam_info_data['f'].values[0]

    # Image LAS match
    img2block = 'img2block.csv'
    img2block_data = pd.read_csv(img2block)
    img_heading = img2block_data['Heading'][3]
    img_roll = img2block_data['Roll'][3]
    img_pitch = img2block_data['Pitch'][3]
    img_Xs = img2block_data['X'][3]
    img_Ys = img2block_data['Y'][3]
    img_Zs = img2block_data['Z'][3]

    img_file = '20121205A//survey1_Cam1_000230.jpg'
    img = cv2.imread(img_file)
    img_b, img_g, img_r = cv2.split(img)
    height = img.shape[0]
    width = img.shape[1]
    las_file = 'las//pt000844.las'
    las = File(las_file)
    las_x = las.x - img_Xs
    las_y = las.y - img_Ys
    las_z = las.z - img_Zs
    # rotation matrix
    # img_heading + cam_heading
    # img_roll + cam_roll
    # img_pitch + cam_pitch
    rotation_matrix = cal_rotation_matrix_xyz(math.radians(img_heading),
                                              math.radians(img_roll),
                                              math.radians(img_pitch))
    pc_matrix = np.concatenate((las_x.reshape(1,-1), las_y.reshape(1,-1), las_z.reshape(1,-1)), axis=0)
    # R_-1 * PC
    temp = np.dot(np.linalg.inv(rotation_matrix), pc_matrix)
    temp_x = temp[0,:]
    temp_y = temp[1,:]
    temp_z = temp[2,:]
    # pc--->img
    pc2img_x = -1 * cam_f * temp_z / temp_x
    pc2img_y = -1 * cam_f * temp_y / temp_x
    pc2img_x = pc2img_x + cam_x0
    pc2img_y = pc2img_y + cam_y0
    # x range: -0.5-height/2 ~ 0.5+height/2
    x_min = (-0.5-height/2) * 0.000006
    x_max = (0.5+height/2) * 0.000006
    # y range: -0.5-width/2 ~ 0.5+width/2
    y_min = (-0.5-width/2) * 0.000006
    y_max = (0.5+width/2) * 0.000006
    result_x = pc2img_x[np.where((pc2img_x>=x_min) & (pc2img_x<=x_max) & (pc2img_y>=y_min) & (pc2img_y<=y_max))]
    result_y = pc2img_y[np.where((pc2img_x>=x_min) & (pc2img_x<=x_max) & (pc2img_y>=y_min) & (pc2img_y<=y_max))]
    result_pixel_x, result_pixel_y = img_xy_length2pixel(result_x, result_y, 0.000006, height, width)
    result_pixel_x[result_pixel_x==0] = 1
    result_pixel_y[result_pixel_y==0] = 1
    img_b[result_pixel_x-1, result_pixel_y-1] = 0
    img_g[result_pixel_x-1, result_pixel_y-1] = 0
    img_r[result_pixel_x-1, result_pixel_y-1] = 255
    result_img = cv2.merge((img_b, img_g, img_r))
    result_img_resize = cv2.resize(result_img, (int(width/4), int(height/4)))
    cv2.imshow('Point Cloud to Image', result_img_resize)
    cv2.waitKey(0)
    cv2.imwrite('PointCloud2Image.jpg',result_img)
    # orientation：general viewing direction and mounting orientation of the camera
    # camera misalignment：orientation difference between IMU and camera. Defined by
    # Heading, Roll, and Pitch misalignment angles