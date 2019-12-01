# Counting the vehicles for each lane
# Author: Jiahao Xia
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import numpy as np
import csv
import datetime
import data

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# Object detection imports
from utils import label_map_util
from utils import visualization_utils as vis_util
import pandas as pd

start = datetime.datetime.now()
os.chdir('D://XJH//RU//course//INTRO TRANS PLANNING//video//')
video_filename = ['VIRB0001.MP4', 'VIRB0001-2.MP4', 'VIRB0001-3.MP4']
video_name = ['VIRB0001', 'VIRB0001-2', 'VIRB0001-3']

# "SSD with Mobilenet" model
MODEL_DIR = 'D://XJH//RU//course//INTRO TRANS PLANNING//VehicleDetection//'
MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'
PATH_TO_CKPT = MODEL_DIR + MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join(MODEL_DIR, 'data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map,
        max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

num_videos = len(video_filename)
# lanes block
xmin = [1103,1142,1229,1163,1103,1057]
xmax = [1239,1363,1540,1409,1199,1157]
ymin = [388,780,777,2957,1287,1513]
ymax = [633,1175,1174,3589,1465,1705]

result = pd.DataFrame(columns={'Location', 'Lane', 'VideoTime', 'VideoMilSeconds'})

for i in range(3):
    print('processing the {}th video......'.format(i + 1))
    m_video = cv2.VideoCapture(video_filename[i])
    # video information
    frame_num = m_video.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_width = int(m_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(m_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps: frames/second
    fps = m_video.get(cv2.CAP_PROP_FPS)

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            for j in range(int(frame_num)):
                (ret, frame) = m_video.read()
                if not ret:
                    print ('end of the video file...')
                    break
                if j % 10 == 0:
                    time_index = 10/fps*int(j/10)
                    for k in range(6):
                        input_frame = frame[xmin[k]:xmax[k],ymin[k]:ymax[k],:]
                        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                        image_np_expanded = np.expand_dims(input_frame, axis=0)

                        # Actual detection.
                        (boxes, scores, classes, num) = \
                            sess.run([detection_boxes, detection_scores,
                                      detection_classes, num_detections],
                                     feed_dict={image_tensor: image_np_expanded})
                        if ((classes==3).any()) | ((classes==6).any()) | ((classes==8).any()):
                            time_index_m, time_index_s = divmod(time_index, 60)
                            time_index_h, time_index_m = divmod(time_index_m, 60)
                            temp_VideoTime = "%d:%02d:%02d" % (time_index_h, time_index_m, time_index_s)
                            temp = pd.DataFrame({'Location': [video_name[i]],
                                                 'Lane': [k+1],
                                                 'VideoTime': [temp_VideoTime],
                                                'VideoMilSeconds': [int(100000*(time_index-int(time_index)))]})
                            result = result.append(temp, ignore_index=True)

result.to_csv('result.csv')
end = datetime.datetime.now()
print(end-start)
