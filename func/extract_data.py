# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 16:49:30 2021

@author: jcv
"""
import pandas as pd

def extract_mts_data(file_path, files, col_dtypes, columns):
    # import csv file
    mts_df = pd.read_csv(os.path.join(file_path,files[0]),skiprows = 5,
                         header = 1)
    # set dataframe columns
    mts_df.columns = columns
    # drop index with units
    mts_df = mts_df.drop(axis = 0, index = 0)
    # set to numeric dtype
    mts_df = mts_df.astype(col_dtypes)
    
    # filter based on trigger value and drop unneeded columns
    cam_trig_df = mts_df[mts_df['trigger'] > 0].drop(['trigger',
                                                      'cam_44','cam_43',
                                                      'trig_arduino'],
                                                     axis = 1)

    # return data at every nth frame    
    return cam_trig_df