# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 12:27:02 2021

@author: jcv
"""
import pandas as pd
import os
import numpy as np
import pyspark
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import BayesianGaussianMixture

def add_features(data_df, img_scale, time_mapping, orientation):
    '''Add features to dataframes for EDA 
    (scaled coordinates, time and stretch ratios)
    
    Args: 
        data_df (dataframe): original dataframe containing basline data for frame
        img_scale (float): mm per pixel image scale
        time_mapping (dict): mapping of frame to test time in seconds
        orientation (string): specify specimen orientation relative to camera FOV
            
    Returns:
        data_df (dataframe): original dataframe with features appended to columns
    '''
    # ------------------------------------------------------------------------
    # ----- add columns with scaled coordinates -----
    # ------------------------------------------------------------------------
    data_df['x_mm'] = data_df['x_pix']*img_scale + data_df['ux']
    data_df['y_mm'] = data_df['y_pix']*img_scale + data_df['uy']
    
    # ------------------------------------------------------------------------
    # ----- calculate axial stretch from Green-Lagrange strain fields ----
    # ------------------------------------------------------------------------
    data_df['lambda_y'] = data_df[['Eyy']].apply(lambda x: np.sqrt(2*x+1))
    
    # ------------------------------------------------------------------------
    # add time to dataframe based on time mapping
    # ------------------------------------------------------------------------
    data_df['time'] = data_df['frame'].map(time_mapping)
    
    # ------------------------------------------------------------------------
    # ----- create in-plane Poisson's ratio feature -----
    # ------------------------------------------------------------------------
    try: 
        if orientation == 'vertical':
            data_df['nu'] = -1*data_df['Exx']/data_df['Eyy']
        elif orientation == 'horizontal':
            data_df['nu'] = -1*data_df['Eyy']/data_df['Exx']
    except:
        print('Specimen orientation not recognized/specified.')
    
    return data_df

def return_frame_df(frame_no, dir_data):
    '''Load pre-processed data (m_process_to_pkl.py) for a single frame 
    
    Args: 
        frame_no (int): frame number to load 
        dir_data (string): directory containing processed data in .pkl format
            
    Returns:
        frame_df (dataframe): dataframe containing full-field data
    '''
    # define file path
    save_filename = 'results_df_frame_' + '{:02d}'.format(frame_no) + '.pkl'
    current_filepath = os.path.join(dir_data,save_filename)
    #load data and add features
    frame_df = pd.read_pickle(current_filepath)
    frame_df['frame'] = frame_no*np.ones((frame_df.shape[0],))
    
    return frame_df

def return_frame_df_spark(frame_no, dir_data, sc):
    '''Load pre-processed data (m_process_to_pkl.py) for a single frame using 
    spark (NEED TO DEBUG)
    
    Args: 
        frame_no (int): frame number to load 
        dir_data (string): directory containing processed data in .pkl format
        sc (object): sql context object previously initiated
            
    Returns:
        frame_df (dataframe): dataframe containing full-field data
    '''
    # define file path
    save_filename = 'results_df_frame_' + '{:02d}'.format(frame_no) + '.pkl'
    current_filepath = os.path.join(dir_data,save_filename)
    #load data and add features
    pickleRdd = sc.pickleFile(current_filepath).collect()
    frame_df = spark.createDataFrame(pickleRdd)
    
    #frame_df = pd.read_pickle(current_filepath)
    frame_df['frame'] = frame_no*np.ones((frame_df.shape[0],))
    
    return frame_df

def return_points_in_all_frames(data_df, last_frame_df):
    '''Return dataframe containing only points in FOV for all frames up to a 
        defined 'last frame' for analysis
    
    Args: 
        data_df (dataframe): dataframe containing measurements for current frame 
        last_frame_df (dataframe): dataframe for last frame in analysis
            
    Returns:
        data_all_df (dataframe): dataframe containing only points in FOV up to 
            last frame specified by last_frame_df
    '''
    # copy data
    data_df_copy = data_df.copy()
    
    # keep only points that appear in last frame
    data_all_df = data_df_copy[data_df_copy.index.isin(last_frame_df.index)]
    
    return data_all_df

def find_points_in_clusters(num_clusters, cluster_ranges, 
                              cluster_var, frame_df):
    '''Find indices of points corresponding to each defined cluster 
        (based on variable passed to function - typ. axial strain)
    
    Args: 
        num_clusters (int): number of clusters to split field data into
        cluster_ranges (dict): upper and lower bounds of cluster ranges
        cluster_var (string): name of variable to split data on
        frame_df (dataframe): dataframe with full-field data for a given frame
            
    Returns:
        indices_dict (dict): lists of point indices corresponding to each cluster
    '''
    # define empty dictionary to store indices for each cluster
    indices_dict = {}
        
    # --- loop through all clusters, find points and plot ---
    for j in range(0,num_clusters):
        if j == num_clusters-1:
            cluster_df = frame_df[
                    frame_df[cluster_var] >= cluster_ranges[j]
                ]   
        else:
            cluster_df = frame_df[(
                    frame_df[cluster_var] >= cluster_ranges[j]
                    ) & (
                    frame_df[cluster_var] < cluster_ranges[j+1]
                )]
        # store index objects in dictionary
        indices_dict[j] = cluster_df.index
        
    return indices_dict

def find_points_in_clusters_ml(num_clusters, frame_df):
    '''Find indices of points corresponding to clusters defined from using a 
        clustering algorithm 
    
    Args: 
        num_clusters (int): number of clusters data divided into
        frame_df (dataframe): dataframe with full-field data for a given frame
            
    Returns:
        indices_dict (dict): lists of point indices corresponding to each cluster
    '''
    indices_dict = {}
        
    # --- loop through all clusters, find points and plot ---
    for j in range(0,num_clusters):
        cluster_df = frame_df[frame_df['cluster'] == j]
        indices_dict[j] = cluster_df.index
        
    return indices_dict

def define_clusters_ml(num_clusters, frame_df, scale_features, cluster_args):
    '''CLuster full-field data using a Bayesian Gaussian Mixture algorithm
    
    Args: 
        num_clusters (int): number of clusters
        frame_df (dataframe): dataframe with full-field data for a given frame
        scale_features (boolean): flag to toggle standard scaling of features
        cluster_args (dict): dictionary of arguments in model definition
            
    Returns:
        frame_df (dataframe): dataframe containing full-field data with cluster
            assignment appended as column
    '''
    
    # define features (coordinates and strain)
    X = frame_df[['x_mm', 'y_mm', 'Eyy', 'Exx']]

    # scale features
    if scale_features:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

    # Bayesian Gaussian mixture model
    bgm = BayesianGaussianMixture(n_components = num_clusters, random_state = 1,
                                  **cluster_args)
    
    if scale_features:
        bgm.fit(Xs)
        y_pred_bgm = bgm.predict(Xs)
    else:
        bgm.fit(X)
        y_pred_bgm = bgm.predict(X)

    # append cluster number to dataframe
    frame_df['cluster'] = y_pred_bgm
    
    return frame_df

def identify_outliers(frame_df, thresholds):
    '''Identify outliers in measurements using square of strain gradient
    
    Args: 
        frame_df (dataframe): dataframe with full-field data for a given frame
        thresholds (array): thresholds for outliers for x and y [x, y]
            
    Returns:
        outliers_df (dataframe): dataframe of full-field data flagged as
            outliers
        filtered_df (dataframe): dataframe of full-field data with 
            outliers excluded
            
    '''
    
    outliers_df = frame_df.copy()
    filtered_df = frame_df.copy()

    outliers_df = outliers_df[(outliers_df['de_dy2'] >= thresholds[1]) 
                              | (outliers_df['de_dx2'] >= thresholds[0])]
    filtered_df = filtered_df[(filtered_df['de_dy2'] < thresholds[1]) 
                              | (filtered_df['de_dx2'] < thresholds[0])]
    
    return outliers_df, filtered_df