# PhD research week1(2019.11.02-2019.11.08)
# Author: Jiahao Xia
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys
import pandas as pd
from itertools import islice

def read_tphoto_imagelist(filepath):
    if not os.path.isfile(filepath):
        sys.exit()
    else:
        imglist = pd.DataFrame(columns=('ImageName', 'Time', 'X', 'Y', 'Z',
                                        'Heading', 'Roll', 'Pitch', 'CameraNum',
                                        'Quality', 'Color'))
        with open(filepath) as fp:
            cnt = 0
            for line in fp:
                line = line.replace('\n','')
                if cnt == 0:
                    print("line {} contents: {}".format(cnt, line))
                if cnt % 8 == 1:
                    ImageName = line.split('=')[1]
                if cnt % 8 == 2:
                    Time = line.split('=')[1]
                    Time = float(Time)
                if cnt % 8 == 3:
                    XYZ = line.split('=')[1]
                    XYZ = XYZ.split(' ')
                    X = float(XYZ[0])
                    Y = float(XYZ[1])
                    Z = float(XYZ[2])
                if cnt % 8 == 4:
                    HRP = line.split('=')[1]
                    HRP = HRP.split(' ')
                    Heading = float(HRP[0])
                    Roll = float(HRP[1])
                    Pitch = float(HRP[2])
                if cnt % 8 == 5:
                    CameraNum = line.split('=')[1]
                    CameraNum = int(CameraNum)
                if cnt % 8 == 6:
                    Quality = line.split('=')[1]
                    Quality = int(Quality)
                if cnt % 8 == 7:
                    Color = line.split('=')[1]
                if (cnt != 0) & (cnt % 8 == 0):
                    temp_imglist = pd.DataFrame({'ImageName': [ImageName], 'Time': [Time], 'X': [X], 'Y': [Y], 'Z': [Z],
                                    'Heading': [Heading], 'Roll': [Roll], 'Pitch': [Pitch],
                                    'CameraNum': [CameraNum], 'Quality': [Quality], 'Color': [Color]})
                    imglist = imglist.append(temp_imglist, ignore_index=True)
                cnt = cnt + 1

        return imglist

def get_block_xy(xy):
# get X Y from the line in tscan_blocklist file
    X = float(xy.split(' ')[0])
    Y = float(xy.split(' ')[1])
    return X, Y

def read_tscan_blocklist(filepath):
    if not os.path.isfile(filepath):
        sys.exit()
    else:
        blocklist = pd.DataFrame(columns=('BlockName', 'X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4'))
        with open(filepath) as fp:
            cnt = 1
            for line in islice(fp, 18, None):
                line = line.replace('\n', '')
                if cnt % 7 == 1:
                    BlockName = line.split(' ')[1]
                if cnt % 7 == 2:
                    X1, Y1 = get_block_xy(line)
                if cnt % 7 == 3:
                    X2, Y2 = get_block_xy(line)
                if cnt % 7 == 4:
                    X3, Y3 = get_block_xy(line)
                if cnt % 7 == 5:
                    X4, Y4 = get_block_xy(line)
                if cnt % 7 == 6:
                    blocklist = blocklist.append({'BlockName': BlockName, 'X1': X1, 'Y1': Y1,
                                                  'X2': X2, 'Y2': Y2, 'X3': X3, 'Y3': Y3, 'X4': X4, 'Y4': Y4},
                                                 ignore_index=True)
                cnt = cnt + 1
        return blocklist

def visualize_block_img(blocklist, imagelist, lasdir):
    # visualize block and img
    # result_block corresponding to the existing las .file
    result_block = pd.DataFrame(columns=('BlockName', 'X_min', 'X_max', 'Y_min', 'Y_max'))
    for file in os.listdir(lasdir):
        if os.path.splitext(file)[1] == '.las':
            print(file)
            temp_block = blocklist.loc[blocklist['BlockName'] == file]
            if temp_block.shape[0] != 0:
                X_min = temp_block.loc[:, ['X1', 'X2', 'X3', 'X4']].min(axis=1).values
                X_max = temp_block.loc[:, ['X1', 'X2', 'X3', 'X4']].max(axis=1).values

                Y_min = temp_block.loc[:, ['Y1', 'Y2', 'Y3', 'Y4']].min(axis=1).values
                Y_max = temp_block.loc[:, ['Y1', 'Y2', 'Y3', 'Y4']].max(axis=1).values

                plt.plot([X_min, X_max, X_max, X_min, X_min], [Y_min, Y_min, Y_max, Y_max, Y_min], 'r-')

                temp = pd.DataFrame({'BlockName': file, 'X_min': X_min, 'X_max': X_max, 'Y_min': Y_min, 'Y_max': Y_max})
                result_block = result_block.append(temp, ignore_index=True)

    # img num
    img_num = imagelist.shape[0]

    img2block = []
    for i in range(img_num):
        temp_ImageName = imagelist['ImageName'][i]
        temp_img_X = imagelist['X'][i]
        temp_img_Y = imagelist['Y'][i]
        plt.plot(temp_img_X, temp_img_Y, 'b*')
        img2block_if = 0
        for j in range(result_block.shape[0]):
            print('img: {} --- block: {}'.format(i, j))
            temp_block_xmin = result_block['X_min'][j]
            temp_block_xmax = result_block['X_max'][j]
            temp_block_ymin = result_block['Y_min'][j]
            temp_block_ymax = result_block['Y_max'][j]
            if (temp_img_X >= temp_block_xmin) & (temp_img_X <= temp_block_xmax) & (temp_img_Y >= temp_block_ymin) & (temp_img_Y <= temp_block_ymax):
                img2block = np.append(img2block, result_block['BlockName'][j])
                img2block_if = 1
        if img2block_if == 0:
            img2block = np.append(img2block, 'NoBlock')
    imagelist['img2block'] = img2block
    plt.show()
    return imagelist

if __name__ == '__main__':
    os.chdir("F://Staten Island//Staten Island")

    imglist_filepath = "20121205A//20121205A_IMAGELIST.iml"
    imglist = read_tphoto_imagelist(imglist_filepath)

    blocklist_filepath = "las//WoolpertSandyMMS_Final.prj"
    blocklist = read_tscan_blocklist(blocklist_filepath)

    img2block = visualize_block_img(blocklist, imglist, 'las//')
    print('***done***')