# TRANS PLANNING --- Video Analysis
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

start = datetime.datetime.now()
# Object detection imports
from utils import label_map_util
from utils import visualization_utils as vis_util

os.chdir('D://XJH//RU//course//INTRO TRANS PLANNING//video//')
video_filename = ['VIRB0001.MP4', 'VIRB0001-2.MP4', 'VIRB0001-3.MP4']

# define the lane detection point
lane_detection_point = [[1209,501],
                        [1331,657],
                        [1499,885],
                        [1307,3621],
                        [1177,3587],
                        [1115,1767],
                        [1141,2227]]

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

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width,
            3)).astype(np.uint8)

num_videos = len(video_filename)
for i in range(1):
    print('processing the {}th video......'.format(i+1))
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

            # while m_video.isOpened():
            for j in range(int(200.0)):
                (ret, frame) = m_video.read()

                if not ret:
                    print ('end of the video file...')
                    break
                # lane1
                xmin = [1103,1142,1229,1163,1103,1057]
                xmax = [1239,1363,1540,1409,1199,1157]
                ymin = [388,780,777,2957,1287,1513]
                ymax = [633,1175,1174,3589,1465,1705]
                frame1 = frame[1103:1239,388:633,:]
                frame2 = frame[1142:1363,780:1175,:]
                frame3 = frame[1229:1540,777:1174,:]
                frame4 = frame[1163:1409,2957:3589,:]
                frame5 = frame[1103:1199,1287:1465,:]
                frame6 = frame[1057:1157,1513:1705,:]
                input_frame = frame1

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(input_frame, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = \
                    sess.run([detection_boxes, detection_scores,
                              detection_classes, num_detections],
                             feed_dict={image_tensor: image_np_expanded})

                (counter, csv_line) = \
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        m_video.get(1),
                        input_frame,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        min_score_thresh=0.2,
                        use_normalized_coordinates=True,
                        line_thickness=4,
                    )

                # input_frame_resize = cv2.resize(input_frame, (int(frame_width/4), int(frame_height/4)))
                cv2.imshow('vehicle detection', input_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            m_video.release()
            # cv2.destroyAllWindows()

end = datetime.datetime.now()
print(end-start)