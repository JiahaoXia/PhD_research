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

x = np.arange(0,200,0.02)
y = np.arange(0,100,0.01)
z = np.repeat(10,len(x))

a = np.array([1,2,3,4,5])