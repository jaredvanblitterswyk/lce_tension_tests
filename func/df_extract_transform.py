# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 12:27:02 2021

@author: jcv
"""
import pandas as pd
import os
import numpy as np

def add_features(data_df, img_scale, time_mapping, orientation):
    # ----------------------------------------------------------------------------
    # ----- add columns with scaled coordinates -----
    # ----------------------------------------------------------------------------
    data_df['x_mm'] = data_df['x_pix']*img_scale + data_df['ux']
    data_df['y_mm'] = data_df['y_pix']*img_scale + data_df['uy']
    
    # ----------------------------------------------------------------------------
    # ----- calculate axial stretch from Green-Lagrange strain fields ----
    # ----------------------------------------------------------------------------
    data_df['lambda_y'] = data_df[['Eyy']].apply(lambda x: np.sqrt(2*x+1))
    
    # ----------------------------------------------------------------------------
    # add time to dataframe based on time mapping
    # ----------------------------------------------------------------------------
    data_df['time'] = data_df['frame'].map(time_mapping)
    
    # ----------------------------------------------------------------------------
    # ----- create in-plane Poisson's ratio feature -----
    # ---------------------------------------------------------------------------
    try: 
        if orientation == 'vertical':
            data_df['nu'] = -1*data_df['Exx']/data_df['Eyy']
        elif orientation == 'horizontal':
            data_df['nu'] = -1*data_df['Eyy']/data_df['Exx']
    except:
        print('Specimen orientation not recognized/specified.')
    
    return data_df

def return_frame_dataframe(frame_no, dir_data):
    # define file path
    save_filename = 'results_df_frame_' + '{:02d}'.format(frame_no) + '.pkl'
    current_filepath = os.path.join(dir_data,save_filename)
    #load data and add features
    frame_df = pd.read_pickle(current_filepath)
    frame_df['frame'] = frame_no*np.ones((frame_df.shape[0],))
    
    return frame_df

def return_points_in_all_frames(data_df, last_frame_df):
    # copy data
    data_df_copy = data_df.copy()
    
    # keep only points that appear in last frame
    data_all_df = data_df_copy[data_df_copy.index.isin(last_frame_df.index)]
    
    return data_all_df

def find_points_in_categories(num_categories, category_ranges, 
                              category_var, frame_df):
    indices_dict = {}
        
    # --- loop through all categories, find points and plot ---
    for j in range(0,num_categories):
        if j == num_categories-1:
            category_band_df = frame_df[
                    frame_df[category_var] >= category_ranges[j]
                ]   
        else:
            category_band_df = frame_df[(
                    frame_df[category_var] >= category_ranges[j]
                    ) & (
                    frame_df[category_var] < category_ranges[j+1]
                )]
        # store index objects in dictionary
        indices_dict[j] = category_band_df.index
        
    return indices_dict