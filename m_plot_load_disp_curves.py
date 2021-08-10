# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 15:55:29 2021

@author: jcv
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% ----- PLOT LOAD VS DISP FOR MULTIPE TESTS -----
# ----------------------------------------------------------------------------

data_dir_root = 'Z:/Experiments/lce_tension/lcei_001/mts_data'
dir_ext = ['lcei_001_007_t04_r00']

filenames = ['em_lcei_001_007_t04_r00.csv']
             
data_dirs = [os.path.join(data_dir_root, f) for f in dir_ext]
file_paths = [os.path.join(data_dirs[ind], f) for ind, f in enumerate(filenames)]

columns = ['time', 'crosshead', 'load', 'trigger', 'cam_44', 'cam43', 'arduino']

# colour and edge colour arrays (hard coded to 10 or less strain bands)
ec = ['#003057', '#00313C', '#9E2A2F', '#623412','#284723',
      '#59315F', '#A45A2A', '#53565A', '#007377', '#453536']
c = ['#BDD6E6', '#8DB9CA', '#F4C3CC', '#D9B48F', '#9ABEAA',
     '#C9B1D0', '#F1C6A7', '#C8C9C7', '#88DBDF', '#C1B2B6']

f = plt.figure(figsize = (7,2.5))
ax = f.add_subplot(1,1,1)

for i in range(0,len(filenames)):
     
    # load data
    df = pd.read_csv(file_paths[i], skiprows = 5, header = 2)
    df.columns = columns
    # select subset of data
    df.iloc[::1000, :]
    
    # plot load vs time (log)
    ax.scatter(x = df[['crosshead']]/25+1, y = df[['load']], 
               s = 0.1, c = c[0], edgecolors = ec[0], linewidths = 0.5 )
    
ax.set_xlabel('$\lambda_y$')
ax.set_ylabel('Load (N)')
#ax.legend(['t05_r00','t05_r01'])
ax.grid(True, alpha = 0.4, zorder = -1)
#ax.set_xscale('log')
plt.tight_layout()
plt.show()

#%% ----- PLOT LOAD VS LOG(TIME) FOR MULTIPE TESTS -----
# ----------------------------------------------------------------------------
dir_ext = ['lcei_001_007_t10_r00',
            'lcei_001_007_t13_r00']

filenames = ['em_lcei_001_007_t10_r00.csv',
             'em_lcei_001_007_t13_r00.csv']

data_dirs = [os.path.join(data_dir_root, f) for f in dir_ext]
file_paths = [os.path.join(data_dirs[ind], f) for ind, f in enumerate(filenames)]

f = plt.figure(figsize = (7,2.5))
ax = f.add_subplot(1,1,1)

for i in range(0,len(filenames)):
    # load data
    df = pd.read_csv(file_paths[i], skiprows = 5, header = 2)
    df.columns = columns
    # select subset of data
    df.iloc[::5000, :]
    
    # plot load vs time (log)
    ax.scatter(x = df[['time']], y = df[['load']] - df.loc[0,'load'], 
               s = 1, c = c[i+1], edgecolors = ec[i+1], linewidths = 0.5 )
    
ax.set_xlabel('Time (s)')
ax.set_ylabel('Load (N)')
ax.legend(['stage 3','stage_3r'])
ax.grid(True, alpha = 0.4, zorder = -1)
ax.set_xscale('log')
plt.tight_layout()
plt.show()

