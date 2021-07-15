# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 10:45:18 2021

@author: jcv
"""
import os
import sys
import csv
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from func.create_eda_plots import create_simple_scatter, plot_boxplot_vs_frame, generate_histogram
from func.df_extract_transform import add_features, return_frame_df, return_points_in_all_frames
from func.extract_data import extract_mts_data
from matplotlib.colors import LinearSegmentedColormap

#%% ----- MAIN SCRIPT -----
# ----------------------------------------------------------------------------
# ----- configure directories, plot colors and constants -----
# ----------------------------------------------------------------------------
# root directory
dir_root = 'Z:/Experiments/lce_tension'
# extensions to access sub-directories
batch_ext = 'lcei_003'
mts_ext = 'mts_data'
sample_ext = '001_t03_r0X'
gom_ext = 'gom_results'
cmap_name = 'lajolla' # custom colormap stored in mpl_styles
frame_map_filename = batch_ext+'_'+sample_ext+'_frame_time_mapping.csv'
mts_columns = ['time', 'crosshead', 'load', 'trigger', 'cam_44', 'cam_43', 'trig_arduino']
mts_col_dtypes = {'time':'float',
              'crosshead':'float', 
              'load':'float',
              'trigger': 'int64',
              'cam_44': 'int64',
              'cam_43': 'int64',
              'trig_arduino': 'int64'}

load_multiple_frames = True # load single frames flag
orientation = 'vertical'
frame_range = 13 # set frame range for plotting histograms
end_frame = 30 # manually define last frame where all points still in FOV
mask_frame = 9 # frame to use to mask points 
post_mask_frame = 25 # frame to compare to mask to determine if strain inc/dec
img_scale = 0.01568 # image scale (mm/pix)

# colour and edge colour arrays (hard coded to 10 or less strain bands)
ec = ['#003057', '#00313C', '#9E2A2F', '#623412','#284723',
      '#59315F', '#A45A2A', '#53565A', '#007377', '#453536']
c = ['#BDD6E6', '#8DB9CA', '#F4C3CC', '#D9B48F', '#9ABEAA',
     '#C9B1D0', '#F1C6A7', '#C8C9C7', '#88DBDF', '#C1B2B6']

# load in colormap
cm_data = np.loadtxt('Z:/Python/mpl_styles/'+cmap_name+'.txt')
custom_map = LinearSegmentedColormap.from_list('custom', cm_data)

# define custom plot style
plt.style.use('Z:/Python/mpl_styles/stg_plot_style_1.mplstyle')

# define full paths to mts and gom data
dir_frame_map = os.path.join(dir_root,batch_ext,mts_ext,batch_ext+'_'+sample_ext)
dir_mts = os.path.join(dir_root,batch_ext,mts_ext,batch_ext+'_'+sample_ext)
dir_gom_results = os.path.join(dir_root,batch_ext,sample_ext,gom_ext)

results_files = [f for f in os.listdir(dir_gom_results) if f.endswith('.pkl')]
num_frames = len(results_files)


# plots to generate
plots_to_generate = ['histogram',
                     'boxplot',
                     'global_stress_strain'
                     ]

#%% ----- LOAD DATA -----
# ----------------------------------------------------------------------------
# ----- load in frame-time mapping file -----
# ----------------------------------------------------------------------------
try:
    frame_map_filepath = os.path.join(dir_frame_map, frame_map_filename)
    frames_list = pd.read_csv(frame_map_filepath)
    #convert to dictionary
    time_mapping = frames_list.iloc[:,2].to_dict()
except:
    print('Frames-time mapping file was not found/loaded.')

# ----------------------------------------------------------------------------
# ----- load in DIC to dataframe -----
# ----------------------------------------------------------------------------
if load_multiple_frames:
    for i in range(1,frame_range+1):
        print('Adding frame: '+str(i))
        save_filename = 'results_df_frame_' + '{:02d}'.format(i) + '.pkl'
        try:
            current_filepath = os.path.join(dir_gom_results,save_filename)
            frame_df = pd.read_pickle(current_filepath)
            
            # add time stamp to frame to allow for sorting later
            frame_df['frame'] = i*np.ones((frame_df.shape[0],))
            
            if i == 1:
                # create empty data frame to store all values from each frame
                data_df = pd.DataFrame(columns = frame_df.columns)
            
            data_df = pd.concat([data_df, frame_df], 
                axis = 0, join = 'outer'
                )
        except:
            print('File not found or loaded succesfully.')
        
    data_df = data_df.dropna(axis = 0)

#%% ----- ADD FEATURES AND DEFINE KEY FRAME DATAFRAMES -----
if load_multiple_frames:
    # add time independent features
    data_df = add_features(data_df, img_scale, time_mapping, orientation)
    # separate frames of interest
    mask_frame_df = data_df[data_df['frame'] == mask_frame]
    first_frame_df = data_df[data_df['frame'] == 1]
    last_frame_df = data_df[data_df['frame'] == end_frame]
    frame_range_df = data_df[data_df['frame'] == frame_range]
  
    # keep only points that appear in last frame
    data_all_df = return_points_in_all_frames(data_df, last_frame_df)

else:
    # define frames of interest
    mask_frame_df = return_frame_df(mask_frame, dir_gom_results)
    first_frame_df = return_frame_df(1, dir_gom_results)
    last_frame_df = return_frame_df(end_frame, dir_gom_results)
    frame_range_df = return_frame_df(frame_range, dir_gom_results)
    
    # add time independent features
    mask_frame_df = add_features(mask_frame_df, img_scale, time_mapping, orientation)
    first_frame_df = add_features(first_frame_df, img_scale, time_mapping, orientation)
    last_frame_df = add_features(last_frame_df, img_scale, time_mapping, orientation)
    frame_range_df = add_features(frame_range_df, img_scale, time_mapping, orientation)

'''
# ----------------------------------------------------------------------------
# ----- calculate temporal change in strain and incr/decr flag -----
# ----------------------------------------------------------------------------
''' 
'''
Procedure:
1) extract first two frames with all points that appear over the full test
2) use this temp df to extract the indices of all points
3) the spacing between points is constant across frames so consider one case
4) use the row spacing between the same points to compute the strain diff
5) store in separate df with flag to indicate if point relaxes or extends w t
'''
'''
# extract points in first two frames  
temp_df = pts_in_all_frames_df[pts_in_all_frames_df['frame'] <= 2 ]

pt_indices = temp_df.index
# find index spacing of one point
indices_single_pt = [i for i, x in enumerate(pt_indices) if x == pt_indices[0]]
pts_period = indices_single_pt[1] - indices_single_pt[0]

pts_in_all_frames_df['dEyy_dt'] = pts_in_all_frames_df['Eyy'].diff(periods = pts_period)  
pts_in_all_frames_df['dsigma_dt'] = pts_in_all_frames_df['stress_mpa'].diff(periods = pts_period)   

# add feature indicating if temporally increasing or decreasing beyond mask frame
# create dictionary mapping indices to value (0 or 1)
mask_df = pts_in_all_frames_df[pts_in_all_frames_df['frame'] == mask_frame]
post_mask_df = pts_in_all_frames_df[pts_in_all_frames_df['frame'] == post_mask_frame]

deyy_dt = mask_df[['Eyy']] - post_mask_df[['Eyy']]
deyy_dt_bool = (deyy_dt > 0).astype(int)

# create separate data frame with coordinates and bool mapping for strain incr.
first_frame_df['dEyy_dt_cat'] = deyy_dt_bool
'''

#%% ----- EXPLORATORY DATA ANALYSIS -----
# ----- plot histogram and box plot of strain for each frame -----
if 'histogram' in plots_to_generate:
    # ----------------------------------------------------------------------------
    # ----- initialize plot vars -----
    # ----------------------------------------------------------------------------
    
    subplot_cols = 6
    subplot_dims = [int(round((frame_range)/subplot_cols,0)), subplot_cols]
    plot_var = 'Eyy'
    
    # compile plot params in dictionary
    plot_params = {'n_bins': 20, 
                   'xlims': [1.05*frame_range_df[plot_var].min(),
                             1.05*frame_range_df[plot_var].max()],
                   'linewidth': 0.5,
                   'grid_alpha': 0.5,
                   'fontsize': 5,
                   'annot_linestyle': '--',
                   'annot_linewidth': 0.4,
                   'annot_fontsize': 4
                   }
    
    if load_multiple_frames: 
        generate_histogram(subplot_dims, plot_var, plot_params, frame_range,
                           load_multiple_frames, dir_gom_results, img_scale, 
                           time_mapping, orientation, ec, c, data_df)
    else:
        generate_histogram(subplot_dims, plot_var, plot_params, frame_range,
                           load_multiple_frames, dir_gom_results, img_scale, 
                           time_mapping, orientation, ec, c)
        
#%% ----- Generate Box Plots -----
# ----------------------------------------------------------------------------
# ----- compute plot specific quantities -----
# ----------------------------------------------------------------------------
'''
data_exx = []
data_eyy = []
data_exy = []

# if loading all frames at once, aggregate strain data into list for plotting
if load_multiple_frames: 
    frame_labels = all_frames_df['frame'].unique().astype(int).astype(str)
    
    # append strains to list for boxplots
    for i in range(0,frame_range):
        data_exx.append(
            np.array(
                all_frames_df[all_frames_df['frame'] == i]['Exx']
                )
            )
        data_eyy.append(
            np.array(
                all_frames_df[all_frames_df['frame'] == i]['Eyy']
                )
            )
        data_exy.append(
            np.array(
                all_frames_df[all_frames_df['frame'] == i]['Exy']
                )
            )
else:
    # data aggregation already complete, just calculate frame labels
    frame_labels = np.linspace(1, frame_range, frame_range).astype(int).astype(str)
    
# ----------------------------------------------------------------------------
# ----- generate box plots -----
# ----------------------------------------------------------------------------
mpl.rcParams['lines.marker']=''
plot_boxplot_vs_frame(data_exx, frame_labels, ylabel = 'Exx')
plot_boxplot_vs_frame(data_eyy, frame_labels, ylabel = 'Eyy')
plot_boxplot_vs_frame(data_exy, frame_labels, ylabel = 'Exy')

#%%
    # calculate measurement area
    avg_width = single_frame_df.groupby('x_pix').first().mean()['width_mm']
    N = single_frame_df.groupby('x_pix').first().shape[0]
    area = avg_width*img_scale*N
'''