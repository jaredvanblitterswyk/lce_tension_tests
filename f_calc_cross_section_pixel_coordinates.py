# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 16:13:49 2021

@author: jcv
"""
import os
import sys
import csv
import pandas as pd
import numpy as np
from numpy import linalg
import matplotlib as mpl
import matplotlib.pyplot as plt

dir_coords = 'Z:/Experiments/lce_tension/lcei_001/007_t01_r00/front'
filename = 'lcei-001-007-t07-r00-front-001.csv'
img_scale = 0.0106 # mm/pix
section_df = pd.read_csv(os.path.join(dir_coords,filename), encoding='UTF-8')

section_df.drop(columns = 'Value', inplace = True)

max_y = section_df.groupby(['X']).max()
min_y = section_df.groupby(['X']).min()

width = max_y - min_y
width_mm = width*img_scale

plt.figure()
plt.scatter(width_mm.index,width_mm)
