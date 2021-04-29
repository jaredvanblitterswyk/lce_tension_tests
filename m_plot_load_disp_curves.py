# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 15:55:29 2021

@author: jcv
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_dir_root = 'Z:/Experiments/lce_tension/lcei_001/mts_data'
dir_ext = ['lcei_001_007_t05_r00',
           'lcei_001_007_t05_r01']

filenames = ['em_lcei_001_007_t05_r00.csv',
             'em_lcei_001_007_t05_r01.csv']
             
data_dirs = [os.path.join(data_dir_root, f) for f in dir_ext]
file_paths = [os.path.join(data_dirs[ind], f) for ind, f in enumerate(filenames)]

columns = ['time', 'crosshead', 'load', 'trigger', 'cam_44', 'cam43', 'arduino']

f = plt.figure(figsize = (5,3))
ax = f.add_subplot(1,1,1)

for i in range(0,len(filenames)):
     
    # load data
    df = pd.read_csv(file_paths[i], skiprows = 5, header = 2)
    df.columns = columns
    # select subset of data
    df.iloc[::10, :]
    
    # plot load vs time (log)
    ax.scatter(x = df[['time']], y = df[['load']], s = 1)
    
ax.set_xlabel('Time (s)')
ax.set_ylabel('Load (N)')
ax.legend(['t05_r00', 't05_r01'])
ax.set_xscale('log')
plt.show()

