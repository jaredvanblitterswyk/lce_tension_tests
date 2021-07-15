# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 13:51:04 2021

@author: jcv
"""
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from func.df_extract_transform import return_frame_df, add_features


def generate_histogram(subplot_dims, plot_var, plot_params, frame_range,
                       load_multiple_frames, dir_gom_results, img_scale, 
                       time_mapping, orientation, ec, c, data_df = None):
    '''Generate histogram for mulitple frames
    
    This function generates a histogram for each frame for a specified target 
    variable. General plot parameters are passed in through a dictionary as 
    well as face and edge color arrays, along with other miscellaneous info
    to facilitate loading the data properly in the case of loading and plotting
    on a frame-by-frame basis. The data_df is an optional pandas DataFrame that
    can be passed if the data has been loaded into memory previously.
    
    '''
    
     # initialize row and column index counters
    row, col = 0, 0
    # ----------------------------------------------------------------------------
    # ----- create figure -----
    # ----------------------------------------------------------------------------
    fig, axs = plt.subplots(subplot_dims[0], subplot_dims[1], sharey=True, tight_layout=True)
        
    # loop through range of frames and generate histogram and box plots
    plot_num = 0
    
    for i in range(1,frame_range):
        # --------- compute axis indices ----------
        row = int(plot_num/(subplot_dims[1]))
        col = plot_num - row*(subplot_dims[1])
        
        # ---------- load data for current frame ----------
        if load_multiple_frames: 
            frame_df = data_df[data_df['frame'] == i]
        else:
            frame_df = return_frame_df(i, dir_gom_results)
            frame_df = add_features(frame_df, img_scale, time_mapping, orientation)
            
        # ---------- generate subplot ----------
        axs[row,col].hist(frame_df[plot_var], edgecolor=ec[0], color = c[0], 
            bins = plot_params['n_bins'], linewidth = plot_params['linewidth'])
        axs[row,col].set_title('Frame: '+ str(i), fontsize = plot_params['fontsize'])
        axs[row,col].set_xlabel(plot_var, fontsize = plot_params['fontsize'])
        axs[row,col].grid(True, alpha = plot_params['grid_alpha'])
        axs[row,col].set_xlim(plot_params['xlims'])
        axs[row,col].tick_params(labelsize = plot_params['fontsize'])
        
        # extract ylims of plot for annotations
        _, max_ylim = plt.ylim()
        
        # add line showning mean of field at each frame
        avg_strain = frame_df[plot_var].mean()
        axs[row,col].axvline(avg_strain,
            color='k', linestyle=plot_params['annot_linestyle'], 
            linewidth = plot_params['annot_linewidth'], marker = ''
            )
        axs[row,col].text(
            avg_strain*1.1, max_ylim*0.8, 
            'Mean: {:.2f}'.format(avg_strain), fontsize = plot_params['annot_fontsize']
            ) 

        # if loading each file individually, delete after use
        del frame_df
        plot_num += 1
        
    plt.show()