# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 16:49:30 2021

@author: jcv
"""
import pandas as pd
import os

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

def extract_load_at_images(mts_df, file_path, current_file, col_dtypes, 
                           columns, keep_frames):
    '''Extract load at images and append to mts dataframe (if created)
    
    Args: 
        mts_df (dataframe): dataframe containing mts measurements 
        filepath (string): path to current results file from gom
        current_file (string): name of current results file from gom (csv)
        col_dtypes (list):  list of data types for columns of mts_df
        columns (list): list of column names
        keep_frames (dataframe): dataframe of frame-time mapping to keep
            
    Returns:
        mts_df (dataframe): dataframe of mts data with current frame appended
    '''
    
    # import csv file
    mts_raw_df = pd.read_csv(os.path.join(file_path,current_file),skiprows = 5,
                         header = 1)
    # set dataframe columns
    mts_raw_df.columns = columns
    # drop index with units
    mts_raw_df = mts_raw_df.drop(axis = 0, index = 0)
    # set to numeric dtype
    mts_raw_df = mts_raw_df.astype(col_dtypes)
    
    # filter based on trigger value and drop unneeded columns
    cam_trig_df = mts_raw_df[mts_raw_df['trigger'] > 0].drop(['time','trigger',
                                                      'cam_44','cam_43',
                                                      'trig_arduino'],
                                                     axis = 1)
    # reset index to match image capture number
    cam_trig_df.reset_index(inplace=True, drop = True)
    
    # select only points where frame number matches subset of images in keep_frames
    cam_trig_subset = cam_trig_df[cam_trig_df.index.isin(list(keep_frames['raw_frame']))]
    
    # append to dataframe
    mts_df = pd.concat([mts_df,cam_trig_subset])
      
    return mts_df