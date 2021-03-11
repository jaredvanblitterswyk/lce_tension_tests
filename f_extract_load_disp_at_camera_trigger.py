# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 09:27:59 2021

@author: jcv
"""
import pandas as pd
import csv
import numpy as np
import os
from os import listdir

# def main():
# define pathname to file
root_dir = 'Z:/Experiments/lce_tension/lcei_001'
mts_dir = 'mts_data'
specimen_dir = 'lcei_001_007_t01_r00'

# save variables to file
file_path = os.path.join(root_dir,mts_dir,specimen_dir)

files = [f for f in os.listdir(file_path) if f.endswith('.csv')]
columns = ['time', 'crosshead', 'load', 'trigger', 'cam_44', 'cam_43', 'trig_arduino']

mts_df = pd.read_csv(os.path.join(file_path,files[0]),skiprows = 5,header = 1)
# set dataframe columns
mts_df.columns = columns
# drop index with units
mts_df = mts_df.drop(axis = 0, index = 0)
# set to numeric dtype
# define dictionary to convert columns
col_dtypes = {'time':'float',
              'crosshead':'float', 
              'load':'float',
              'trigger': 'int64',
              'cam_44': 'int64',
              'cam_43': 'int64',
              'trig_arduino': 'int64'}

mts_df = mts_df.astype(col_dtypes)

cam_trig_df = mts_df[mts_df['trigger'] > 0] 
cam_trig_df.astype(col_dtypes,copy=False).dtypes
cam_trig_df.plot.scatter(x='crosshead', y='load', s=1)

dic_sync_load_xhead_df = cam_trig_df.iloc[::5,:]

dic_sync_load_xhead_df.plot.scatter(x='crosshead', y='load', s=1)


# if __name__ == '__main__':
#     main()
# else:
#     # return variable
#     print('Run as function.')