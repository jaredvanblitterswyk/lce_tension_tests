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
from func.mts_extract_data import extract_load_at_images
from func.compute_fields import (compute_R, 
                                 calculateEijRot,
                                 mask_interp_region,
                                 interp_and_calc_strains,
                                 calc_xsection
                                 )
#%% ---- MAIN SCRIPT ----
# ----- configure directories -----
# root directory
dir_root = 'Z:/Experiments/lce_tension'
dir_root_local = 'C:/Users/jcv/Documents'
# extensions to access sub-directories
batch_ext = 'lcei_003'
mts_ext = 'mts_data'
sample_ext = '009_t02_r05'
gom_ext = 'gom_results'
time_map_ext = 'frame_time_mapping'

# define full paths to mts and gom data
try:
    dir_xsection = os.path.join(dir_root,batch_ext,sample_ext)
    dir_mts = os.path.join(dir_root,batch_ext,mts_ext,batch_ext+'_'+sample_ext)
    dir_gom_results = os.path.join(dir_root,batch_ext,sample_ext,gom_ext)
    dir_frame_map = os.path.join(dir_root,batch_ext,mts_ext,batch_ext+'_'+sample_ext,time_map_ext)
except:
    print('One of setup directories does not exist.')

# ----- define constants -----
spec_id = batch_ext+'_'+sample_ext # full specimen id
Nx, Ny = 2448, 2048 # pixel resolution in x, y axis
img_scale = 0.02729 # mm/pix
t = 1 # thickness of sample [mm]
orientation = 'vertical'
cmap_name = 'lajolla' # custom colormap stored in mpl_styles
xsection_filename = batch_ext+'_'+sample_ext+'_section_coords.csv'
mask_side_length = 0.5 # max side length of triangles in DeLauny triangulation
mts_columns = ['time', 'crosshead', 'load', 'trigger', 'cam_44', 'cam_43', 'trig_arduino']
mts_col_dtypes = {'time':'float',
              'crosshead':'float', 
              'load':'float',
              'trigger': 'int64',
              'cam_44': 'int64',
              'cam_43': 'int64',
              'trig_arduino': 'int64'}
coord_trans_applied = False

# ---- calculate quantities and collect files to process ---
# collect all csv files in gom results directory
files_gom = [f for f in os.listdir(dir_gom_results) if f.endswith('.csv')]
# collect mts data file
files_mts = [f for f in os.listdir(dir_mts) if f.endswith('.csv')]
# collect frame-time conversion files
files_frames = [f for f in os.listdir(dir_frame_map) if f.endswith('.csv')]

# create vector of x dimensions
x_vec = np.linspace(0,Nx-1,Nx) # vector - original x coords (pix)
y_vec = np.linspace(Ny-1,0,Ny) # vector - original y coords (pix)

xx_pix,yy_pix = np.meshgrid(x_vec,y_vec) # matrix of x and y coordinates (pix)
xx,yy = xx_pix*img_scale, yy_pix*img_scale # matrix of x and y coordinates (mm)

# crop coordinates to contain specimen (reduce interp comp. cost)
xc1, xc2 = 850, 1625 # col indices to crop coordinates
yc1, yc2 = 0, Ny # row indices to crop coordinates
Nxc, Nyc = xc2-xc1, yc2-yc1 # dims of cropped coordinates for reshaping

# -- cropped coordinates to interpolate onto ---
xx_crop = xx[yc1:yc2,xc1:xc2]
yy_crop = yy[yc1:yc2,xc1:xc2]

xx_pix_crop = xx_pix[yc1:yc2,xc1:xc2]
yy_pix_crop = yy_pix[yc1:yc2,xc1:xc2]

# calculate scaled spacing between points in field
dx, dy = img_scale, img_scale

# calculate width of specimen at each pixel location
try:
    width_mm = calc_xsection(
        dir_xsection, xsection_filename, img_scale, orientation
        )
except: 
    print('No file containing cross-section coordinates found.')

try:
    frames_list = pd.read_csv(os.path.join(dir_frame_map,files_frames[0]))
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

#%%
# ---- run processing -----  
disp_labels = ['ux', 'uy', 'uz']
strain_labels = ['Exx', 'Eyy', 'Exy']

processing_params = {}
processing_params['mask_side_length'] = mask_side_length
processing_params['spacing'] = [dx, dy]
processing_params['image_scale'] = img_scale
processing_params['image_dims'] = [Nx, Ny]
processing_params['coord_trans_applied'] = coord_trans_applied

for i in range(0,len(mts_df)):
    start_time = time.time()
    # extract frame number and display
    frame_no = files_gom[i][28:-10] # lcei_001_006_t02_r00
    #frame_no = files_gom[i][35:-10] 
    print('Processing frame:' + str(frame_no))
    
    # compute interpolated strains and displacements in reference coordinates
    disp, Eij, Rij, area_mask, triangle_mask = interp_and_calc_strains(
        dir_gom_results, files_gom[i], processing_params, disp_labels, 
        strain_labels, xx_crop, yy_crop)
    
    # compute strain gradient to find outliers
    de_dy, de_dx = np.gradient(Eij['Eyy'])
    
    # assemble results in data frame
    outputs_df = pd.DataFrame()
    
    for component in disp_labels:
        outputs_df[component] = np.reshape(disp.get(component),(Nxc*Nyc,))
    
    for component in strain_labels:
        outputs_df[component] = np.reshape(Eij.get(component),(Nxc*Nyc,))

    outputs_df['R'] = np.reshape(Rij,(Nxc*Nyc,))
    outputs_df['de_dy'] = np.reshape(de_dy,(Nxc*Nyc,))
    outputs_df['de_dx'] = np.reshape(de_dx,(Nxc*Nyc,))
    
    # square and take inverse to exaggerate outliers
    outputs_df['de_dx2'] = outputs_df['de_dx'].apply(lambda x: x**2)
    outputs_df['de_dy2'] = outputs_df['de_dy'].apply(lambda x: x**2)
    
    # ----- compile results into dataframe -----
    # concatenate to create one output dataframe
    results_df = pd.concat([coords_df,outputs_df], axis = 1, join = 'inner')
    # drop points with nans
    results_df = results_df.dropna(axis=0, how = 'any')

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
    
    save_filename = 'results_df_frame_' + '{:02d}'.format(int(frame_no)) + '.pkl'
    # table = pa.Table.from_pandas(results_df)
    # pq.write_table(table, save_filename)
    #results_df.to_parquet(os.path.join(dir_root_local,save_filename),
    #                      engine='pyarrow', index=True)
    results_df.to_pickle(os.path.join(dir_root_local,save_filename))
    print("--- %s seconds ---" % (time.time() - start_time))