import os
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from laspy.file import File
os.chdir('F://Staten Island//Staten Island//')

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

x_axis = np.array([np.arange(0,1,0.01),np.repeat(0,100),np.repeat(0,100)])
y_axis = np.array([np.repeat(0,100),np.arange(0,1,0.01),np.repeat(0,100)])
z_axis = np.array([np.repeat(0,100),np.repeat(0,100),np.arange(0,1,0.01)])
result = np.concatenate((x_axis,y_axis,z_axis),axis=1)

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

fig = plt.figure()
ax1 = fig.add_subplot(1,2,1, projection='3d')
# ax1.axes(projection='3d')
ax1.scatter(result[0,:],result[1,:],result[2,:],color='r')

ax2 = fig.add_subplot(1,2,2, projection='3d')
rotation_result = np.dot(rotation_matrix, result)
ax2.scatter(rotation_result[0,:],rotation_result[1,:],rotation_result[2,:],color='g')