# Translation and Rotation
# Author: Jiahao Xia
import os
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from laspy.file import File
from mpl_toolkits.mplot3d import Axes3D
import math

os.chdir('F://Staten Island//Staten Island//')

def las_visualization(las_data, title, color, outfile):
    """
    Visualization of .las
    :param las_data: .las
    :param title: the title of the figure
    :param color: the color of the point cloud
    :param outfile: where the figure is saved
    :return: no return
    """
    x = las_data.x
    y = las_data.y
    z = las_data.z
    # figure dpi
    fig = plt.figure(dpi=120)
    ax = fig.add_subplot(111, projection='3d')
    plt.title(title)
    ax.scatter(x, y, z, c=color, marker='.', s=1, linewidth=0, alpha=1, cmap='spectral')
    # ax.axis('scaled')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.savefig(outfile)

def camera_info_read(cam_file):
    """
    get the heading, roll, pitch, x0, y0, f
    :param cam_file:
    :return: pd.Dataframe
    """
    cam_info = pd.read_csv(cam_file)
    cam_heading = cam_info['AttitudeC_heading'].values
    cam_roll = cam_info['AttitudeC_roll'].values
    cam_pitch = cam_info['AttitudeC_pitch'].values
    cam_X = cam_info['PrincipalP_X'].values * 0.000006
    cam_Y = cam_info['PrincipalP_Y'].values * 0.000006
    cam_Z = cam_info['PrincipalP_Z'].values * 0.000006
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

# Image LAS match
img2block = 'img2block.csv'
img2block_data = pd.read_csv(img2block)
img_num = img2block_data.shape[0]

# cameras information files
cam_info = ['camera1_cal.csv', 'camera2_cal.csv', 'camera3_cal.csv', 'camera4_cal.csv']
cam_info_data = pd.DataFrame(columns=['heading', 'roll', 'pitch', 'x0', 'y0', 'f'])
for i in range(4):
    temp = camera_info_read(cam_info[i])
    cam_info_data = cam_info_data.append(temp)

# for i in range(img_num)
for i in range(1):
    temp_image_file = img2block_data['ImageName'][i]
    temp_las_file = img2block_data['img2block'][i]
    img_heading = img2block_data['Heading'][i]
    img_roll = img2block_data['Roll'][i]
    img_pitch = img2block_data['Pitch'][i]
    img_Xs = img2block_data['X'][i]
    img_Ys = img2block_data['Y'][i]
    img_Zs = img2block_data['Z'][i]

    if temp_las_file != 'NoBlock':
        temp_image = cv2.imread('20121205A//' + temp_image_file)
        temp_las = File('las//' + temp_las_file)
        # las_visualization(temp_las, temp_las_file, 'r', temp_las_file[0:-4]+'.jpg')
        temp_cam_no = int(temp_image_file.split('_')[1][-1])
        cam_heading = (cam_info_data['heading'].values)[temp_cam_no-1]
        cam_roll = (cam_info_data['roll'].values)[temp_cam_no-1]
        cam_pitch = (cam_info_data['pitch'].values)[temp_cam_no-1]
        # principal point information
        cam_x0 = (cam_info_data['x0'].values)[temp_cam_no-1]
        cam_y0 = (cam_info_data['y0'].values)[temp_cam_no-1]
        cam_f = (cam_info_data['f'].values)[temp_cam_no-1]

        heading = img_heading + cam_heading
        roll = img_roll + cam_roll
        pitch = img_pitch + cam_pitch

        rotation_matrix = cal_rotation_matrix(math.radians(heading),
                                              math.radians(roll),
                                              math.radians(pitch))
