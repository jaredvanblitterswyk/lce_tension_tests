# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 16:46:54 2021

@author: jcv
"""
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from func.df_extract_transform import return_frame_df, add_features

def create_simple_scatter(plot_vars, plot_params, plot_frame_range,
                          load_multiple_frames, dir_results, img_scale, 
                          time_mapping, orientation, ec, c, data_df = None):
    '''Generate boxplots for one variable as a function of frame number
    
    Args: 
        plot_vars (dict): dictionary of x and y plot variables
        plot_params (dict): dictionary of parameters to customize plot
        plot_frame_range (array): min and max frame numbers to plot
        load_multiple_frames (bool): flag to process in batch or frame-by-frame
        dir_gom_results (str): dic results dictionary
        img_scale (float): mm/pixel scale for images
        time_mapping (dict): map frame number to test time
        orientation (str): orientation of sample in field of view
        ec (array): list of possible marker edge colours
        c (array): list of possible face colours
        data_df (dataframe, optional): pre-loaded results from all frames in
            one data structure
            
    Returns:
        Figure
        
    Notes: 
        Hard-coded to group by frame

    '''
    # ------------------------------------------------------------------------
    # ----- create figure -----
    # ------------------------------------------------------------------------
    f = plt.figure(figsize = plot_params['figsize'])
    ax = f.add_subplot(1,1,1)
    
    # ----- load data -----
    # ----- if preloaded, use groupby method to get data -----
    if load_multiple_frames:
        x = data_df.groupby('frame')[plot_vars['x']].mean()
        y = data_df.groupby('frame')[plot_vars['y']].mean()
    else:
        x = []
        y = []
        # ---------- load data for current frame ----------
        for i in range(plot_frame_range[0], plot_frame_range[1]+1):
            frame_df = return_frame_df(i, dir_results)
            frame_df = add_features(frame_df, img_scale, time_mapping, orientation)    
            
            x.append(frame_df[plot_vars['x']].mean())
            y.append(frame_df[plot_vars['y']].mean())
            
    # ---------- add data ----------   
    ax.scatter(x = x, y = y, s = plot_params['m_size'], 
               c = c[0], edgecolors = ec[0], 
               linewidths = plot_params['linewidth'])
    ax.set_xlabel(plot_params['xlabel'], fontsize = plot_params['fontsize'])
    ax.set_ylabel(plot_params['ylabel'], fontsize = plot_params['fontsize'])
    ax.grid(True, alpha = plot_params['grid_alpha'], zorder=0)
    ax.tick_params(labelsize = plot_params['fontsize'])
    if plot_params['log_x']:
        ax.set_xscale('log')
    if plot_params['tight_layout']:
        plt.tight_layout()
        
    plt.show()
    
def generate_boxplot_vs_frame(plot_var, plot_params, plot_frame_range,
                              load_multiple_frames, dir_results, img_scale, 
                              time_mapping, orientation, data_df = None):
    '''Generate boxplots for one variable as a function of frame number
    
    Args: 
        plot_var (str): variable to plot
        plot_params (dict): dictionary of parameters to customize plot
        plot_frame_range (array): min and max frame numbers to plot
        load_multiple_frames (bool): flag to process in batch or frame-by-frame
        dir_gom_results (str): dic results dictionary
        img_scale (float): mm/pixel scale for images
        time_mapping (dict): map frame number to test time
        orientation (str): orientation of sample in field of view
        data_df (dataframe, optional): pre-loaded results from all frames in
            one data structure
            
    Returns:
        Figure

    '''
    # genenerate frame labels
    frame_labels = [f for f in range(plot_frame_range[0],plot_frame_range[1])]
    
    # ------------------------------------------------------------------------
    # ----- create figure -----
    # ------------------------------------------------------------------------
    f = plt.figure(figsize = plot_params['figsize'])
    ax = f.add_subplot(1,1,1) 
    
    # ---------- load data for current frame ----------
    for i in range(plot_frame_range[0], plot_frame_range[1]):
        if load_multiple_frames: 
            frame_df = data_df[data_df['frame'] == i]
        else:
            frame_df = return_frame_df(i, dir_results)
            frame_df = add_features(frame_df, img_scale, time_mapping, orientation)    
            
        # ---------- add data ----------        
        ax.boxplot(frame_df[plot_var].values, positions = [i], 
                   showfliers = plot_params['showfliers'])
        ax.set_xticklabels(frame_labels)
        ax.set_xlabel(plot_params['xlabel'], fontsize = plot_params['fontsize'])
        ax.set_ylabel(plot_params['ylabel'], fontsize = plot_params['fontsize'])
        ax.set_xlim(plot_params['xlims'])
        ax.tick_params(labelsize = plot_params['fontsize'])
        ax.grid(True, alpha = plot_params['grid_alpha'], zorder = 0)
        plt.tight_layout()
        
    plt.show()
    
def generate_histogram(subplot_dims, plot_var, plot_params, plot_frame_range,
                       load_multiple_frames, dir_results, img_scale, 
                       time_mapping, orientation, ec, c, data_df = None):
    '''Generate histogram for mulitple frames in separate subplots
    
    Args: 
        subplot_dims (array): number of subplots in rows and columns
        plot_var (str): variable to plot
        plot_params (dict): dictionary of parameters to customize plot
        plot_frame_range (array): min and max frame numbers to plot
        load_multiple_frames (bool): flag to process in batch or frame-by-frame
        dir_gom_results (str): dic results dictionary
        img_scale (float): mm/pixel scale for images
        time_mapping (dict): map frame number to test time
        orientation (str): orientation of sample in field of view
        ec (array): list of possible marker edge colours
        c (array): list of possible face colours
        data_df (dataframe, optional): pre-loaded results from all frames in
            one data structure
            
    Returns:
        Figure

    '''
    
    # initialize row and column index counters
    row, col = 0, 0
    # ------------------------------------------------------------------------
    # ----- create figure -----
    # ------------------------------------------------------------------------
    fig, axs = plt.subplots(subplot_dims[0], subplot_dims[1], sharey=True, 
                            tight_layout=True)
        
    # loop through range of frames and generate histogram and box plots
    plot_num = 0
    
    for i in range(plot_frame_range[0],plot_frame_range[1]+1):
        # --------- compute axis indices ----------
        row = int(plot_num/(subplot_dims[1]))
        col = plot_num - row*(subplot_dims[1])
        
        # ---------- load data for current frame ----------
        if load_multiple_frames: 
            frame_df = data_df[data_df['frame'] == i]
        else:
            frame_df = return_frame_df(i, dir_results)
            frame_df = add_features(frame_df, img_scale, time_mapping, orientation)
            
        # ---------- generate subplot ----------
        axs[row,col].hist(frame_df[plot_var], edgecolor=ec[0], color = c[0], 
            bins = plot_params['n_bins'], linewidth = plot_params['linewidth'])
        axs[row,col].set_title('Frame: '+ str(i), fontsize = plot_params['fontsize'])
        axs[row,col].set_xlabel(plot_var, fontsize = plot_params['fontsize'])
        axs[row,col].grid(True, alpha = plot_params['grid_alpha'], zorder = 0)
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