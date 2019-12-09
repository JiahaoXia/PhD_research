# The Transformation of Translation and Rotation
# From Image to Point Cloud
# Author: Jiahao Xia
import os
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from laspy.file import File
from mpl_toolkits.mplot3d import Axes3D
import math
import laspy
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

def pointcloud_visualization(x, y, z, title, color, outfile):
    """
    Visualization of point cloud
    :param x: x coordinate of point cloud
    :param y: y coordinate of point cloud
    :param z: z coordinate of point cloud
    :param title: the title of the figure
    :param color: the color of the point cloud
    :param outfile: where the figure is saved
    :return: no return
    """
    # figure dpi
    fig = plt.figure(dpi=120)
    ax = fig.add_subplot(111, projection='3d')
    plt.title(title)
    ax.scatter(x, y, z, marker='.', s=1, linewidth=0, alpha=1, cmap='spectral')
    # ax.axis('scaled')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # plt.savefig(outfile)

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

    rotation_matrix = cal_rotation_matrix(math.radians(img_heading),
                                          math.radians(img_roll),
                                          math.radians(img_pitch))

    img_file = '20121205A//survey1_Cam1_000230.jpg'
    img_matrix = img_read_processing(img_file, cam_x0, cam_y0, cam_f, 0.000006)

    # projection image--->point cloud
    img2pc = np.dot(rotation_matrix, img_matrix)
    pointcloud_visualization(x=img2pc[0,:], y=img2pc[1,:], z=img2pc[2,:],
                            title='Cam1_000230_pc', color='r', outfile='Cam1_000230_pc.jpg')
    # write point cloud into .las file
    header = laspy.header.Header()
    outfile = laspy.file.File('Cam1_000230_pc.las', mode='w', header=header)
    outfile.X = img2pc[0,:]
    outfile.Y = img2pc[1,:]
    outfile.Z = img2pc[2,:]
    outfile.close()
    # las_file = 'las//pt000844.las'
    # las = File(las_file)

