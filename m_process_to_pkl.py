# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 17:55:43 2021

@author: jcv
"""
import os
import csv

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.font_manager as font_manager
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import csv
import pandas as pd
import numpy as np
import math as m
import processing_params as udp
from func.mts_extract_data import extract_load_at_images
from func.compute_fields import (compute_R, 
                                 calculateEijRot,
                                 mask_interp_region,
                                 interp_and_calc_strains,
                                 calc_xsection
                                 )
#%% ---- INITIALIZE DIRECTORIES -----

# define full paths to mts and gom data
try:
    dir_xsection = os.path.join(udp.dir_root, udp.batch_ext, udp.sample_ext)
    dir_mts = os.path.join(dir_root,batch_ext,mts_ext,batch_ext+'_'+sample_ext)
    dir_gom_results = os.path.join(dir_xsection, udp.gom_ext)
    dir_frame_map = os.path.join(dir_mts, udp.time_map_ext)
except:
    print('One of setup directories does not exist.')

#%% ----- LOAD DATA AND COMPUTE FEATURES -----
# ---- calculate quantities and collect files to process ---
# collect all csv files in gom results directory
files_gom = [f for f in os.listdir(dir_gom_results) if f.endswith('.csv')]
# collect mts data file
files_mts = [f for f in os.listdir(dir_mts) if f.endswith('.csv')]
# collect frame-time conversion files
files_frames = [f for f in os.listdir(dir_frame_map) if f.endswith('.csv')]

# create vector of x dimensions
x_vec = np.linspace(0, udp.Nx - 1, udp.Nx) # vector - original x coords (pix)
y_vec = np.linspace(udp.Ny - 1, 0, udp.Ny) # vector - original y coords (pix)

xx_pix,yy_pix = np.meshgrid(x_vec,y_vec) # matrix of x and y coordinates (pix)
xx,yy = xx_pix*udp.img_scale, yy_pix*udp.img_scale # matrix of x and y coordinates (mm)

# crop coordinates to contain specimen (reduce interp comp. cost)
# dims of cropped coordinates for reshaping
Nxc, Nyc = udp.xc2 - udp.xc1, udp.yc2 - udp.yc1 

# -- cropped coordinates to interpolate onto ---
xx_crop = xx[udp.yc1:udp.yc2, udp.xc1:udp.xc2]
yy_crop = yy[udp.yc1:udp.yc2, udp.xc1:udp.xc2]

xx_pix_crop = xx_pix[udp.yc1:udp.yc2, udp.xc1:udp.xc2]
yy_pix_crop = yy_pix[udp.yc1:udp.yc2, udp.xc1:udp.xc2]

# calculate scaled spacing between points in field
dx, dy = udp.img_scale, udp.img_scale

# calculate width of specimen at each pixel location
try:
    width_mm = calc_xsection(
        dir_xsection, xsection_filename, img_scale, orientation
        )
except: 
    print('No file containing cross-section coordinates found.')

try:
    frames_list = pd.read_csv(os.path.join(dir_frame_map, files_frames[0]))
    # keep only image number to extract from raw mts file
    keep_frames = frames_list[['raw_frame']]
    
    for i in range(0,len(files_mts)):
        current_file = files_mts[i]
        if i == 0:
            mts_df = pd.DataFrame(columns = ['crosshead', 'load'])
        
        mts_df = extract_load_at_images(mts_df, dir_mts, current_file,
                                        mts_col_dtypes, mts_columns, 
                                        keep_frames
                                        )
    # reset index
    mts_df.reset_index(drop=True, inplace = True)
    
except:
    print('No file containing mts measurements found.')

# assemble results dataframe
coords_df = pd.DataFrame()
coords_df['x_pix'] = np.reshape(xx_pix_crop,(Nxc*Nyc,))
coords_df['y_pix'] = np.reshape(yy_pix_crop,(Nxc*Nyc,))

#%% ----- PROCESS FIELD DATA FROM GOM ----- 

# assemble processing parameter dictionary to pass to function 
processing_params = {}
processing_params['mask_side_length'] = udp.mask_side_length
processing_params['spacing'] = [dx, dy]
processing_params['image_scale'] = udp.img_scale
processing_params['image_dims'] = [udp.Nx, udp.Ny]
processing_params['coords_origin_center'] = udp.coords_origin_center

# loop through each frame individually
for i in range(0,len(mts_df)):
    start_time = time.time()
    # extract frame number and display
    frame_no = files_gom[i][28:-10]
    #frame_no = files_gom[i][35:-10] 
    print('Processing frame:' + str(frame_no))
    
    # compute interpolated strains and displacements in reference coordinates
    disp, Eij, Rij, area_mask, triangle_mask = interp_and_calc_strains(
        dir_gom_results, files_gom[i], processing_params, udp.disp_labels, 
        udp.strain_labels, xx_crop, yy_crop)
       
    # assemble results in data frame
    outputs_df = pd.DataFrame()
    # displacement components
    for component in disp_labels:
        outputs_df[component] = np.reshape(disp.get(component),
                                           (udp.Nxc*udp.Nyc,))
    
    # strain components
    for component in strain_labels:
        outputs_df[component] = np.reshape(Eij.get(component),
                                           (udp.Nxc*udp.Nyc,))

    outputs_df['R'] = np.reshape(Rij,(udp.Nxc*udp.Nyc,))
    
    # ----- compile results into dataframe -----
    # concatenate to create one output dataframe
    results_df = pd.concat([coords_df,outputs_df], axis = 1, join = 'inner')
    # drop points with nans
    results_df = results_df.dropna(axis=0, how = 'any')
    
    # add cross section width, area and stress features
    try:
        if orientation == 'horizontal':
            results_df['width_mm'] = results_df['x_pix'].apply(
                lambda x: width_mm.loc[x][0] if x in width_mm.index else np.nan
                )
        else: 
            results_df['width_mm'] = results_df['y_pix'].apply(
                lambda x: width_mm.loc[x][0] if x in width_mm.index else np.nan
                )    
        # assign cross-section area
        results_df['area_mm2'] = results_df['width_mm'].apply(lambda x: x*t)
        # assign width-averaged stress
        results_df['stress_mpa'] = mts_df.iloc[int(frame_no),1] / results_df['area_mm2']
    except: 
        print('No cross-section coordinates loaded - excluding width, area and stress from dataframe.')
    
    # drop rows with no cross-section listed
    #results_df = results_df.dropna(axis=0, how = 'any')
    
    # create filename for output pkl file
    save_filename = 'results_df_frame_' + '{:02d}'.format(int(frame_no)) + '.pkl'
    
    # save results dataframe to pkl file
    if udp.save_file_local:
        results_df.to_pickle(os.path.join(dir_root_local,save_filename))
    else:
        results_df.to_pickle(os.path.join(dir_root,save_filename))
        
    print("--- %s seconds ---" % (time.time() - start_time))