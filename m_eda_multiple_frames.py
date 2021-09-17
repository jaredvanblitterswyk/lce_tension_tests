# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 10:45:18 2021

@author: jcv
"""
import os
import sys
import sys
sys.path.append('Z:/Python/tension_test_processing')
sys.path.append(os.path.join(sys.path[-1],'func'))
import csv
import findspark
import pyspark
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import user_defined_processing_parameters as udp
from func.plot_field_contour_save import *
from func.create_eda_plots import (create_simple_scatter, 
                                   histogram_vs_frame, 
                                   boxplot_vs_frame,
                                   var_clusters_vs_time_subplots,
                                   overlay_pts_on_sample,
                                   compressibility_check_clusters,
                                   var_vs_time_clusters_same_axis,
                                   norm_stress_strain_rates_vs_time)
from func.df_extract_transform import (add_features, 
                                       return_frame_df, 
                                       return_frame_df_spark,
                                       return_points_in_all_frames,
                                       find_points_in_clusters,
                                       find_points_in_clusters_ml,
                                       define_clusters_ml)
from func.mts_extract_data import extract_mts_data

#%% ----- MAIN SCRIPT -----
plt.style.use('Z:/Python/mpl_styles/stg_plot_style_1.mplstyle')
plt.close('all')
#%% ----- LOAD DATA -----
# ----------------------------------------------------------------------------
# ----- load in frame-time mapping file -----
# ----------------------------------------------------------------------------
try:
    frame_map_filepath = os.path.join(udp.dir_frame_map, udp.frame_map_filename)
    frames_list = pd.read_csv(frame_map_filepath)
    #convert to dictionary
    time_mapping = frames_list.iloc[:,2].to_dict()
except:
    print('Frames-time mapping file was not found/loaded.')

# ----------------------------------------------------------------------------
# ----- load in DIC to dataframe -----
# ----------------------------------------------------------------------------
if udp.load_multiple_frames:
    for i in range(udp.frame_min, udp.frame_max+1):
        print('Adding frame: '+str(i))
        save_filename = 'results_df_frame_' + '{:02d}'.format(i) + '.pkl'
        try:
            current_filepath = os.path.join(udp.dir_results, save_filename)
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
print('Load data for key frames and add features')
if udp.load_multiple_frames:
    # add time independent features
    data_df = add_features(data_df, udp.img_scale, time_mapping, udp.orientation)
    # separate frames of interest
    mask_frame_df = data_df[data_df['frame'] == udp.mask_frame]
    first_frame_df = data_df[data_df['frame'] == udp.frame_min]
    first_frame_rel_df = data_df[data_df['frame'] == udp.frame_rel_min]
    last_frame_df = data_df[data_df['frame'] == udp.frame_max]
    frame_end_df = data_df[data_df['frame'] == udp.end_frame]
  
    # keep only points that appear in last frame
    data_all_df = return_points_in_all_frames(data_df, last_frame_df)

else:
    # define frames of interest
    mask_frame_df = return_frame_df(udp.mask_frame, udp.dir_results)
    first_frame_df = return_frame_df(udp.frame_min, udp.dir_results)
    first_frame_rel_df = return_frame_df(udp.frame_rel_min, udp.dir_results)
    last_frame_df = return_frame_df(udp.frame_max, udp.dir_results)
    frame_end_df = return_frame_df(udp.end_frame, udp.dir_results)
    
    # add time independent features
    mask_frame_df = add_features(mask_frame_df, udp.img_scale, time_mapping, udp.orientation)
    first_frame_df = add_features(first_frame_df, udp.img_scale, time_mapping, udp.orientation)
    first_frame_rel_df = add_features(first_frame_rel_df, udp.img_scale, time_mapping, udp.orientation)
    last_frame_df = add_features(last_frame_df, udp.img_scale, time_mapping, udp.orientation)
    frame_end_df = add_features(frame_end_df, udp.img_scale, time_mapping, udp.orientation)
       
#%% ----- EXPLORATORY DATA ANALYSIS -----
# ----------------------------------------------------------------------------
# ----- print list of plots that will be generated -----
print('-----------------------------------------------')
print('Plots to be generated:')
for p in udp.plt_to_generate:
    print(p)
    
print('-----------------------------------------------')

# define clusters using ML
if udp.clusters_ml:
    last_frame_df = define_clusters_ml(udp.num_clusters, last_frame_df, 
                                       udp.scale_features, udp.cluster_args)
    
    xx = np.array(last_frame_df[['x_mm']])
    yy = np.array(last_frame_df[['y_mm']])
    zz = np.array(last_frame_df[['cluster']]) 
    
    plot_params_cluster = {
        'figsize': (2,4),
        'xlabel': 'x (mm)', 
        'ylabel': 'y (mm)', 
        'm_size': 0.01, 
        'grid_alpha': 0.5,
        'dpi': 300, 'cmap': udp.custom_map,
        'xlims': [24, 40],
        'ylims': [0, 32],#m.ceil(Ny*img_scale)],
        'tight_layout': True, 
        'hide_labels': False, 
        'show_fig': True,
        'save_fig': False
        }   
    plot_params_cluster['vmin'] = 0
    plot_params_cluster['vmax'] = udp.num_clusters
    plot_params_cluster['var_name'] = 'Cluster No.'
    plot_field_contour_save(xx, yy, zz, plot_params_cluster, udp.frame_max)
    
    plot_params_cluster['vmin'] = 0
    plot_params_cluster['vmax'] = 0.4
    plot_params_cluster['var_name'] = 'Eyy'
    zz = np.array(last_frame_df[['Eyy']])
    plot_field_contour_save(xx, yy, zz, plot_params_cluster, udp.frame_max)

if 'global_stress_strain' in udp.plt_to_generate and udp.load_multiple_frames:
    # import analysis parameters
    anlys_ss = udp.anlys_params_glob_ss
    
    x_glob_ss = data_df.groupby('frame')[anlys_ss['plot_vars']['x']].mean()
    y_glob_ss = data_df.groupby('frame')[anlys_ss['plot_vars']['y']].mean() 

if 'var_clusters_vs_time_subplots' in udp.plt_to_generate:
    # ----- initialize analysis variables -----
    anlys_vcts = udp.anlys_params_var_clusters_subplots
    
    if udp.clusters_ml:
        cluster_indices = find_points_in_clusters_ml(udp.num_clusters, 
                                                             last_frame_df)
        
    else:
        # calculate strain range bounds
        max_cluster_band = round(mask_frame_df[anlys_vcts['cat_var']].quantile(0.98),2)
        min_cluster_band = round(mask_frame_df[anlys_vcts['cat_var']].min(),2)
        
        cluster_ranges = np.linspace(min_cluster_band, max_cluster_band, 
                                      anlys_vcts['num_clusters']) 
    
        # find indices of points on sample belonging to each cluster
        cluster_indices = find_points_in_clusters(anlys_vcts['num_clusters'], 
                                                     cluster_ranges, 
                                                     anlys_vcts['cat_var'], 
                                                     mask_frame_df
                                                     )
    
    # add clusters to analysis parameters dictionary
    anlys_vcts['cluster_indices'] = cluster_indices
    anlys_vcts['cluster_ranges'] = cluster_ranges
    
if 'compressibility_check' in udp.plt_to_generate:
    # ----- initialize analysis variables -----
    anlys_cc = udp.anlys_params_comp_check
    
    if udp.clusters_ml:
        cluster_indices = find_points_in_cluster(udp. num_clusters, 
                                                             last_frame_df)
        
    else:
        # calculate strain range bounds
        max_cluster_band = round(mask_frame_df[anlys_cc['cat_var']].quantile(0.98),2)
        min_cluster_band = round(mask_frame_df[anlys_cc['cat_var']].min(),2)
        
        cluster_ranges = np.linspace(min_cluster_band, max_cluster_band, 
                                      anlys_cc['num_clusters']) 
    
        # find indices of points on sample belonging to each cluster
        cluster_indices = find_points_in_clusters(anlys_cc['num_clusters'], 
                                                     cluster_ranges, 
                                                     anlys_cc['cat_var'], 
                                                     mask_frame_df
                                                     )
    
    # add clusters to analysis parameters dictionary
    anlys_cc['cluster_indices'] = cluster_indices
    anlys_cc['cluster_ranges'] = cluster_ranges
    
if 'overlay_pts_on_sample_var' in udp.plt_to_generate:
    # ----- initialize analysis variables -----
    anlys_opsv = udp.anlys_params_pts_overlay_var
    
    if udp.clusters_ml:
        cluster_indices = find_points_in_clusters_cluster(udp. num_clusters, 
                                                             last_frame_df)
        
    else:
        # calculate strain range bounds
        max_cluster_band = round(mask_frame_df[anlys_opsv['cat_var']].quantile(0.98),2)
        min_cluster_band = round(mask_frame_df[anlys_opsv['cat_var']].min(),2)
        
        cluster_ranges = np.linspace(min_cluster_band, max_cluster_band, 
                                      anlys_opsv['num_clusters']) 
    
        # find indices of points on sample belonging to each cluster
        cluster_indices = find_points_in_clusters(anlys_opsv['num_clusters'], 
                                                     cluster_ranges, 
                                                     anlys_opsv['cat_var'], 
                                                     mask_frame_df
                                                     )
    
    # add clusters to analysis parameters dictionary
    anlys_opsv['cluster_indices'] = cluster_indices
    anlys_opsv['cluster_ranges'] = cluster_ranges
    
if 'overlay_pts_on_sample_relative' in udp.plt_to_generate:
    # ----- initialize analysis variables -----
    anlys_opsr = udp.anlys_params_pts_overlay_relative
    # define series representing change in cluster var between frames    
    cluster_ranges = [-np.inf, 0]
        
    diff_df = pd.DataFrame()
    
    diff_df[anlys_opsr['cat_var']] = last_frame_df[anlys_opsr['var_interest']] - first_frame_rel_df[anlys_opsr['var_interest']]
            
    cluster_indices = find_points_in_clusters(anlys_opsr['num_clusters'],
                                                 cluster_ranges, 
                                                 anlys_opsr['cat_var'],
                                                 diff_df)
    
    # add clusters to analysis parameters dictionary
    anlys_opsr['cluster_indices'] = cluster_indices
    anlys_opsr['cluster_ranges'] = cluster_ranges
    
if 'var_vs_time_clusters_same_axis' in udp.plt_to_generate:
    # ----- initialize analysis variables -----
    anlys_vtcsa = udp.anlys_params_var_vs_time_clusters_sa
    
    # define series representing change in cluster var between frames    
    cluster_ranges = [-np.inf, 0]
        
    diff_df = pd.DataFrame()
    
    diff_df[anlys_vtcsa['cat_var']] = last_frame_df[anlys_vtcsa['y_var']] - first_frame_rel_df[anlys_vtcsa['y_var']]
            
    cluster_indices = find_points_in_clusters(anlys_vtcsa['num_clusters'],
                                                 cluster_ranges, 
                                                 anlys_vtcsa['cat_var'],
                                                 diff_df)
    
    # add clusters to analysis parameters dictionary
    anlys_vtcsa['cluster_indices'] = cluster_indices
    anlys_vtcsa['cluster_ranges'] = cluster_ranges
    
    
if 'norm_stress_strain_rates_vs_time' in udp.plt_to_generate:
    # ----- initialize analysis variables -----
    anlys_nssrt = udp.anlys_params_norm_ss_rates_vs_time
    
    # define series representing change in cluster var between frames    
    cluster_ranges = [-np.inf, 0]
        
    diff_df = pd.DataFrame()
    
    diff_df[anlys_nssrt['cat_var']] = last_frame_df[anlys_nssrt['y_var']] - first_frame_rel_df[anlys_nssrt['y_var']]
            
    cluster_indices = find_points_in_clusters(anlys_nssrt['num_clusters'],
                                                 cluster_ranges, 
                                                 anlys_nssrt['cat_var'],
                                                 diff_df)
    
    # add clusters to analysis parameters dictionary
    anlys_nssrt['cluster_indices'] = cluster_indices
    anlys_nssrt['cluster_ranges'] = cluster_ranges
    
# ----------------------------------------------------------------------------
# --- Plot cluster points overlaid on sample ---
# ----------------------------------------------------------------------------
if 'overlay_pts_on_sample_var' in udp.plt_to_generate:
    # ----- initialize plot vars -----
    plt_opsv = udp.plt_params_pts_overlay_var
    
    # update plot parameters dictionary with plot limits
    plt_opsv['xlims'] = [0.95*round(first_frame_df['x_mm'].min(),1),
                     1.05*round(first_frame_df['x_mm'].max(),1)]
    plt_opsv['ylims'] = [0.95*round(first_frame_df['y_mm'].min(),1),
                     1.05*round(first_frame_df['y_mm'].max(),1)]
           
    # ----- create figure -----
    fig_opsv = plt.figure(figsize = plt_opsv['figsize'])
    ax_opsv = fig_opsv.add_subplot(1,1,1)
    
    overlay_pts_on_sample(plt_opsv, first_frame_df, mask_frame_df, 
                          anlys_opsv, udp.img_scale, ax_opsv) 
    
if 'overlay_pts_on_sample_relative' in udp.plt_to_generate:
    # ----- initialize plot vars -----
    plt_opsr = udp.plt_params_pts_overlay_relative
    
    # update plot parameters dictionary with plot limits
    plt_opsr['xlims'] = [0.95*round(first_frame_df['x_mm'].min(),1),
                         1.05*round(first_frame_df['x_mm'].max(),1)]
    plt_opsr['ylims'] = [0.95*round(first_frame_df['y_mm'].min(),1),
                         1.05*round(first_frame_df['y_mm'].max(),1)]
    
    # ----- create figure -----
    fig_opsr = plt.figure(figsize = plt_opsr['figsize'])
    ax_opsr = fig_opsr.add_subplot(1,1,1)
    
    overlay_pts_on_sample(plt_opsr, first_frame_df, mask_frame_df, 
                          anlys_opsr, udp.img_scale, ax_opsr) 
    
#%%      
# ----------------------------------------------------------------------------
# ---------- Plot figures requiring iteration through time ----------
# ----------------------------------------------------------------------------
for i in range(udp.plt_frame_range[0],udp.plt_frame_range[1]+1):
    print('Processing frame: '+ str(i)+ ' ...')
    # ---------- load data for current frame ----------
    if udp.load_multiple_frames: 
        frame_df = data_df[data_df['frame'] == i]
    else:
        frame_df = return_frame_dataframe(i, udp.dir_results)
        frame_df = add_features(frame_df, udp.img_scale, time_mapping, 
                                udp.orientation)
    # ------------------------------------------------------------------------
    # ----- plot histogram for each frame -----
    # ------------------------------------------------------------------------
    if 'histogram' in udp.plt_to_generate:
        plt_hist = udp.plt_params_histogram
        anlys_hist = udp.anlys_params_histogram
        
        # update plot parameters dictionary with plot limits
        plt_hist['xlims'] = [1.05*last_frame_df[anlys_hist['plot_var']].min(),
                             1.05*last_frame_df[anlys_hist['plot_var']].max()]
        
        if i == udp.plt_frame_range[0]:
            # initialize row and column index counters
            plot_num = 0
            # ----- create figure -----
            fig_h, axs_h = plt.subplots(plt_hist['subplot_dims'][0], 
                                      plt_hist['subplot_dims'][1], 
                                      sharey=True)
        
        histogram_vs_frame(frame_df, anlys_hist, plt_hist, udp.plt_frame_range,
                           plot_num, i, axs_h)

        plot_num += 1
    
    # ------------------------------------------------------------------------
    # ----- Generate Box Plots -----
    # ------------------------------------------------------------------------
    # compile plot params in dictionary
    if 'boxplot' in udp.plt_to_generate:
        
        # ----- import/initialize plot vars -----
        mpl.rcParams['lines.marker'] = ''
        
        plt_bp = udp.plt_params_boxplot
        anlys_bp = udp.anlys_params_boxplot

        # ----- create figure -----
        if i == udp.plt_frame_range[0]:
            fig_bp = plt.figure(figsize = plt_bp['figsize'])
            ax_bp = fig_bp.add_subplot(1,1,1)

        boxplot_vs_frame(frame_df, anlys_bp, plt_bp, udp.plt_frame_range, i, ax_bp)
        
    # ------------------------------------------------------------------------
    # ----- store data for plotting global variables -----
    # ------------------------------------------------------------------------    
    if 'global_stress_strain' in udp.plt_to_generate and not udp.load_multiple_frames:
        if i == udp.plt_frame_range[0]:       
            x_glob_ss = []
            y_glob_ss = []
            
            anlys_ss = udp.anlys_params_glob_ss
            
        x_glob_ss.append(frame_df[anlys_ss['x']].mean())
        y_glob_ss.append(frame_df[anlys_ss['y']].mean())
    
    # ------------------------------------------------------------------------
    # --- Plot clusters of data vs frame based on value in a specified frame ---
    # ------------------------------------------------------------------------
    if 'var_clusters_vs_time_subplots' in udp.plt_to_generate:
        # ----- import/initialize plot properties -----
        plt_vcts = udp.plt_params_var_clusters_subplots
        
        # update plot parameters dictionary with plot limits
        plt_vcts['xlims'] = [1.05*round(first_frame_df[anlys_vcts['x_var']].min(),1),
                             1.1*round(last_frame_df[anlys_vcts['x_var']].max(),1)]
        plt_vcts['ylims'] = [1.05*round(first_frame_df[anlys_vcts['y_var']].min(),1),
                             1.05*round(last_frame_df[anlys_vcts['y_var']].max(),1)]
        
        # create figure and initialize arrays for field average data
        if i == udp.plt_frame_range[0]:
            field_avg_var = []
            field_avg_x = []
            
            # ----- create figure -----
            fig_vcts, ax_vcts = plt.subplots(plt_vcts['subplot_dims'][0], 
                                     plt_vcts['subplot_dims'][1], 
                                     figsize = plt_vcts['figsize'], 
                                     sharey=True)
        
        # collect field average quantities for comparison
        field_avg_var.append(frame_df[anlys_vcts['y_var']].mean())
        field_avg_x.append(frame_df[anlys_vcts['x_var']].mean())
                
        var_clusters_vs_time_subplots(frame_df, anlys_vcts, plt_vcts, 
                                      udp.plt_frame_range, time_mapping, 
                                      field_avg_var, field_avg_x, i, ax_vcts)
        
    if 'compressibility_check' in udp.plt_to_generate:
        # ----- initialize plot vars -----
        plt_cc = udp.plt_params_comp_check
                
        # update plot parameters dictionary with plot limits
        plt_cc['xlims'] = [1.2*round(first_frame_df[anlys_cc['x_var']].min(),1),
                           1.2*round(last_frame_df[anlys_cc['x_var']].quantile(0.995),2)]
        plt_cc['ylims'] = [1.2*round(first_frame_df[anlys_cc['y_var']].quantile(0.995),2),
                           1.2*round(last_frame_df[anlys_cc['y_var']].quantile(0.001),2)]
        
        # ----- create figure -----
        if i == udp.plt_frame_range[0]:
            fig_cc = plt.figure(figsize = plt_cc['figsize'])
            ax_cc = fig_cc.add_subplot(1,1,1)
        
        compressibility_check_clusters(frame_df, anlys_cc, plt_cc, 
                                       udp.plt_frame_range, i, ax_cc)
        
    if 'var_vs_time_clusters_same_axis' in udp.plt_to_generate:
        # ----- initialize plot vars -----
        plt_vtcsa = udp.plt_params_var_vs_time_clusters_sa
        
        # update plot parameters dictionary with plot limits
        plt_vtcsa['xlims'] = [1.2*round(first_frame_df[anlys_vtcsa['x_var']].min(),1),
                             1.2*round(last_frame_df[anlys_vtcsa['x_var']].quantile(0.995),2)]
        plt_vtcsa['ylims'] = [1.2*round(first_frame_df[anlys_vtcsa['y_var']].quantile(0.001),2),
                             1.2*round(last_frame_df[anlys_vtcsa['y_var']].quantile(0.995),2)]
    
        # ----- create figure -----
        if i == udp.plt_frame_range[0]:
            fig_vtcsa = plt.figure(figsize = plt_vtcsa['figsize'])
            ax_vtcsa = fig_vtcsa.add_subplot(1,1,1)
            
        var_vs_time_clusters_same_axis(frame_df, anlys_vtcsa, plt_vtcsa,  
                                  udp.plt_frame_range, i, ax_vtcsa)
        
    if 'norm_stress_strain_rates_vs_time' in udp.plt_to_generate:        
        # assign placeholder objects to store data from all frames
        if i == udp.plt_frame_range[0]:
            cluster_series = {}
            cluster_series['y1_0'] = []
            cluster_series['y1_1'] = []
            cluster_series['y2_0'] = []
            cluster_series['y2_1'] = []
            x_series = []
            
            # extract times for each frame
            time_ = []
            for key in time_mapping.keys():
                time_.append(time_mapping[key])
            dt = np.diff(time_[udp.plt_frame_range[0]:udp.plt_frame_range[1]+1])
            anlys_nssrt['dt'] = dt
        
        # ----- compile data for each cluster and each frame -----           
        for j in range(0, anlys_nssrt['num_clusters']):
            cluster_df = frame_df[frame_df.index.isin(anlys_nssrt['cluster_indices'][j].values)]
            if j == 0:
                #extract mean of x variable only once per frame
                x = cluster_df.groupby(
                anlys_nssrt['x_var'])[anlys_nssrt['x_var']].mean()
                x_series.append(x.values[0])
            
            # extract mean of all points in cluster
            y = cluster_df.groupby(
                anlys_nssrt['x_var'])[anlys_nssrt['y_var']].mean()
            y2 = cluster_df.groupby(
                anlys_nssrt['x_var'])[anlys_nssrt['y_var_2']].mean()
            
            # append to list
            cluster_series['y1_'+str(j)].append(y.values[0])
            cluster_series['y2_'+str(j)].append(y2.values[0])
                    
plt.tight_layout()       
plt.show()

# ----------------------------------------------------------------------------
# ----- plot figures that don't require iteratively loading -----
# ----------------------------------------------------------------------------
if 'global_stress_strain' in udp.plt_to_generate:
    # ----- import/initialize parameters -----
    plt_ss = udp.plt_params_glob_ss
       
    # ----- create figure -----
    fig7 = plt.figure(figsize = plt_ss['figsize'])
    ax7 = fig7.add_subplot(1,1,1)
    
    create_simple_scatter(x_glob_ss, y_glob_ss, plt_ss, udp.plt_frame_range, ax7)

if 'norm_stress_strain_rates_vs_time' in udp.plt_to_generate:
    # ----- initialize plot vars -----
    plt_nssrt = udp.plt_params_norm_ss_rates_vs_time
       
    # ----- initialize plot vars -----
    # define plot parameters dictionary
    plt_nssrt['xlims'] = [1.2*round(first_frame_df[anlys_nssrt['x_var']].min(),1),
                         1.2*round(last_frame_df[anlys_nssrt['x_var']].quantile(0.995),2)]
    plt_nssrt['ylims'] = [0.0001, 1.05]
    
    # add plot variables to analysis_params dictionary
    anlys_nssrt['x_series'] = x_series
    anlys_nssrt['cluster_series'] = cluster_series
    
    # ----- create figure -----
    fig_nssrt = plt.figure(figsize = plt_nssrt['figsize'])
    ax_nssrt = fig_nssrt.add_subplot(1,1,1)

    norm_stress_strain_rates_vs_time(anlys_nssrt, plt_nssrt,
                                     udp.plt_frame_range, i, ax_nssrt)