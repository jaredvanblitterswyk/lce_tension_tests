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
from func.create_eda_plots import (create_simple_scatter, generate_histogram, 
                                   generate_boxplot_vs_frame,
                                   plot_var_classes_over_time,
                                   overlay_pts_on_sample)
from func.df_extract_transform import (add_features, return_frame_dataframe, 
                                       return_points_in_all_frames,
                                       find_points_in_categories)
from func.mts_extract_data import extract_mts_data
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

load_multiple_frames = False # load single frames flag
orientation = 'vertical'
frame_max = 13 # max frame to consider
frame_min = 1 # min frame to plot
frame_range = frame_max - frame_min
plot_frame_range = [frame_min, frame_max] # range of frames to plot
end_frame = 30 # manually define last frame where all points still in FOV
mask_frame = 9 # frame to use to mask points 
post_mask_frame = 25 # frame to compare to mask to determine if strain inc/dec
img_scale = 0.01568 # image scale (mm/pix)

# colour and edge colour arrays (hard coded to 10 or less strain bands)
ec = ['#917265', '#896657', '#996f71', '#805a66', '#453941',
      '#564e5c', '#32303a']
c = ['#d1c3bd', '#ccb7ae', '#b99c9d', '#a6808c', '#8b7382',
     '#706677', '#565264']

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
plots_to_generate = ['overlay_pts_on_sample', 
                     'next_plot'
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
    for i in range(frame_min,frame_max+1):
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
    first_frame_df = data_df[data_df['frame'] == frame_min]
    last_frame_df = data_df[data_df['frame'] == frame_max]
    frame_end_df = data_df[data_df['frame'] == end_frame]
  
    # keep only points that appear in last frame
    data_all_df = return_points_in_all_frames(data_df, last_frame_df)

else:
    # define frames of interest
    mask_frame_df = return_frame_dataframe(mask_frame, dir_gom_results)
    first_frame_df = return_frame_dataframe(frame_min, dir_gom_results)
    last_frame_df = return_frame_dataframe(frame_max, dir_gom_results)
    frame_end_df = return_frame_dataframe(end_frame, dir_gom_results)
    
    # add time independent features
    mask_frame_df = add_features(mask_frame_df, img_scale, time_mapping, orientation)
    first_frame_df = add_features(first_frame_df, img_scale, time_mapping, orientation)
    last_frame_df = add_features(last_frame_df, img_scale, time_mapping, orientation)
    frame_end_df = add_features(frame_end_df, img_scale, time_mapping, orientation)

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
    # ------------------------------------------------------------------------
    # ----- initialize plot vars -----
    # ------------------------------------------------------------------------
    
    subplot_cols = 6
    subplot_dims = [int(round((frame_range)/subplot_cols,0)), subplot_cols]
    plot_var = 'Eyy'
    
    # compile plot params in dictionary
    plot_params = {'n_bins': 20, 
                   'xlims': [1.05*frame_max_df[plot_var].min(),
                             1.05*frame_max_df[plot_var].max()],
                   'ylabel': plot_var,
                   'grid_alpha': 0.5,
                   'fontsize': 5,
                   'annot_linestyle': '--',
                   'annot_linewidth': 0.4,
                   'annot_fontsize': 4
                   }
    
    if load_multiple_frames: 
        generate_histogram(subplot_dims, plot_var, plot_params, plot_frame_range,
                           load_multiple_frames, dir_gom_results, img_scale, 
                           time_mapping, orientation, ec, c, data_df)
    else:
        generate_histogram(subplot_dims, plot_var, plot_params, plot_frame_range,
                           load_multiple_frames, dir_gom_results, img_scale, 
                           time_mapping, orientation, ec, c)
        
#%% ----- Generate Box Plots -----
# ----------------------------------------------------------------------------
# ----- generate for single variable -----
# ----------------------------------------------------------------------------
# compile plot params in dictionary
if 'boxplot' in plots_to_generate:
    
    mpl.rcParams['lines.marker']=''
    
    plot_var = 'Eyy'
    plot_params = {'figsize': (4,2), 
                   'xlims': [0, frame_max+1],
                   'xlabel': 'Frame number',
                   'ylabel': plot_var,
                   'linewidth': 0.5,
                   'grid_alpha': 0.5,
                   'fontsize': 8,
                   'showfliers': False
                   }
                   
    if load_multiple_frames:
        generate_boxplot_vs_frame(plot_var, plot_params, plot_frame_range,
                              load_multiple_frames, dir_gom_results, img_scale, 
                              time_mapping, orientation, data_df)
    else:
        generate_boxplot_vs_frame(plot_var, plot_params, plot_frame_range,
                              load_multiple_frames, dir_gom_results, img_scale, 
                              time_mapping, orientation, data_df)
        
#%% ----- Plot global stress-strain curve -----
# ----------------------------------------------------------------------------
if 'global_stress_strain' in plots_to_generate:
    
    plot_params = {'figsize': (4,2),
               'm_size': 2,
               'linewidth': 0.5,
               'xlabel': 'Time (s)',
               'ylabel': 'Eng. Stress (MPa)',
               'tight_layout': True,
               'grid_alpha': 0.5,
               'fontsize': 8,
               'log_x': True}
    
    plot_vars = {'x': 'time',
                 'y': 'stress_mpa'}
     
    if load_multiple_frames:
        create_simple_scatter(plot_vars, plot_params, plot_frame_range,
                          load_multiple_frames, dir_gom_results, img_scale, 
                          time_mapping, orientation, ec, c, data_df)
    else:
        create_simple_scatter(plot_vars, plot_params, plot_frame_range,
                          load_multiple_frames, dir_gom_results, img_scale, 
                          time_mapping, orientation, ec, c)
        
#%% --- Plot classes of data vs frame based on value in a specified frame ---
if 'scatter_var_categories' in plots_to_generate:
    # ------------------------------------------------------------------------
    # ----- initialize plot vars -----
    # ------------------------------------------------------------------------
    # define categorical variables
    num_categories = 6
    y_var = 'Eyy'
    x_var = 'time'
    category_var = 'Eyy'
    
    # define analysis parameters dictionary
    analysis_params = {'x_var': x_var,
                       'y_var': y_var,
                       'cat_var': category_var,
                       'samples': 8000,
                       'mask_frame': mask_frame}
    
    # define plot parameters dictionary
    plot_params = {'figsize': (5,4),
               'xlabel': 'Time (s)',
               'ylabel': 'Eng. Stress (MPa)',
               'tight_layout': True,
               'grid_alpha': 0.5,
               'm_size': 2,
               'm_alpha': 0.4,
               'fontsize': 5,
               'linewidth': 0.5,
               'annot_linestyle': '--',
               'linestyle': '-',
               'xlims': [1.05*round(first_frame_df[x_var].min(),1),
                         1.1*round(last_frame_df[x_var].max(),1)],
               'ylims': [1.05*round(first_frame_df[y_var].min(),1),
                         1.05*round(last_frame_df[y_var].max(),1)],
               'log_x': True
               }
    
    subplot_cols = 3
    subplot_dims = [int(round((num_categories)/subplot_cols,0)), subplot_cols]
    
    # calculate strain range bounds
    max_category_band = round(mask_frame_df[category_var].quantile(0.98),2)
    min_category_band = round(mask_frame_df[category_var].min(),2)
    
    category_ranges = np.linspace(min_category_band, max_category_band, 
                                  num_categories)
    
    # find indices of points on sample belonging to each category
    category_indices = find_points_in_categories(num_categories, category_ranges, 
                                  category_var, mask_frame_df)
    
    if load_multiple_frames:
        plot_var_classes_over_time(subplot_dims, analysis_params, plot_params, 
                                   num_categories, category_indices, 
                                   category_ranges, plot_frame_range, 
                                   load_multiple_frames, dir_gom_results, 
                                   img_scale, time_mapping, orientation, ec,
                                   c, data_df)
    else:
        plot_var_classes_over_time(subplot_dims, analysis_params, plot_params, 
                                   num_categories, category_indices, 
                                   category_ranges, plot_frame_range, 
                                   load_multiple_frames, dir_gom_results, 
                                   img_scale, time_mapping, orientation, ec, c)
        
#%% ----- overlay physical locaitons of clusters on sample -----
if 'overlay_pts_on_sample' in plots_to_generate:
        
    # ------------------------------------------------------------------------
    # ----- initialize plot vars -----
    # ------------------------------------------------------------------------
    num_categories = 6
    category_var = 'Eyy'
    # compute aspect ratio of sample to set figure size
    width = first_frame_df['x_mm'].max() - first_frame_df['x_mm'].min()
    height = first_frame_df['y_mm'].max() - first_frame_df['y_mm'].min()
    axis_buffer = 1
    fig_width = 2.5
    fig_height = height*axis_buffer/width*fig_width
    
    # define plot parameters dictionary
    plot_params = {'figsize': (fig_width,fig_height),
               'xlabel': 'x (mm)',
               'ylabel': 'y (mm)',
               'ref_c': '#D0D3D4',
               'ref_ec': '#D0D3D4',
               'ref_alpha': 0.3,
               'cluster_alpha': 1.0,
               'tight_layout': False,
               'axes_scaled': True,
               'grid_alpha': 0.5,
               'm_size': 2,
               'm_legend_size': 7,
               'm_alpha': 0.4,
               'fontsize': 5,
               'linewidth': 0,
               'linestyle': '-',
               'xlims': [0.95*round(first_frame_df['x_mm'].min(),1),
                         1.05*round(first_frame_df['x_mm'].max(),1)],
               'ylims': [0.95*round(first_frame_df['y_mm'].min(),1),
                         1.05*round(first_frame_df['y_mm'].max(),1)],
               }
    
     # calculate strain range bounds
    max_category_band = round(mask_frame_df[category_var].quantile(0.85),2)
    min_category_band = round(mask_frame_df[category_var].min(),2)
    
    category_ranges = np.linspace(min_category_band, max_category_band, 
                                  num_categories)
    
    # find indices of points on sample belonging to each category
    category_indices = find_points_in_categories(num_categories, category_ranges, 
                                  category_var, mask_frame_df)

    if load_multiple_frames:
        overlay_pts_on_sample(plot_params, mask_frame, num_categories, 
                              category_indices, category_ranges, 
                              plot_frame_range, load_multiple_frames, 
                              dir_gom_results, img_scale, c, data_df)
    else:
        overlay_pts_on_sample(plot_params, mask_frame, num_categories, 
                              category_indices, category_ranges, 
                              plot_frame_range, load_multiple_frames, 
                              dir_gom_results, img_scale, c)