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
from func.df_extract_transform import return_frame_dataframe, add_features

def create_simple_scatter(plot_vars, plot_params, plot_frame_range,
                          load_multiple_frames, dir_results, img_scale, 
                          time_mapping, orientation, ec, c, data_df = None):
    '''Generate boxplots for one variable as a function of frame number
    
    Args: 
        plot_vars (dict): dictionary of x and y plot variables
        plot_params (dict): dictionary of parameters to customize plot
        plot_frame_range (array): min and max frame numbers to plot
        load_multiple_frames (bool): flag to process in batch or frame-by-frame
        dir_results (str): dic results dictionary
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
            frame_df = return_frame_dataframe(i, dir_results)
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
        dir_results (str): dic results dictionary
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
            frame_df = return_frame_dataframe(i, dir_results)
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
        dir_results (str): dic results dictionary
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
            frame_df = return_frame_dataframe(i, dir_results)
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
    
def plot_var_classes_over_time(subplot_dims, analysis_params, plot_params, 
                               num_categories, category_indices, 
                               category_ranges, plot_frame_range, 
                               load_multiple_frames, dir_results, img_scale, 
                               time_mapping, orientation, ec, c, data_df = None):
    '''Generate scatter plots of values from points on the sample belonging to
    categories defined based on an input variable and mask frame
    
    Args: 
        subplot_dims (array): number of subplots in rows and columns
        analysis_params (dict): variables used for plot on x,y and categories
        plot_params (dict): dictionary of parameters to customize plot
        num_categories (int): number of categories used to cluster points
        category_indices (dict): contains index series of points corresponding
            to each category
        category_ranges (array): bounds on ranges used to cluster sample
        plot_frame_range (array): min and max frame numbers to plot
        load_multiple_frames (bool): flag to process in batch or frame-by-frame
        dir_results (str): dic results dictionary
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
    # ------------------------------------------------------------------------
    # ----- create figure -----
    # ------------------------------------------------------------------------
    fig, axs = plt.subplots(subplot_dims[0], subplot_dims[1], 
                            figsize=plot_params['figsize'], sharey=True)
     
    for i in range(plot_frame_range[0],plot_frame_range[1]+1):
        field_avg_var = []
        field_avg_x = []
        # ---------- load data for current frame ----------
        if load_multiple_frames: 
            frame_df = data_df[data_df['frame'] == i]
        else:
            frame_df = return_frame_dataframe(i, dir_results)
            frame_df = add_features(frame_df, img_scale, time_mapping, 
                                    orientation)
            
        field_avg_var.append(frame_df[analysis_params['y_var']].mean())
        field_avg_x.append(frame_df[analysis_params['x_var']].mean())
        
        for j in range(0,num_categories):
            row = int(j/(subplot_dims[1]))
            col = j - row*(subplot_dims[1])
            
            category_df = frame_df[frame_df.index.isin(category_indices[j].values)]
            
            if category_df.shape[0] > analysis_params['samples']:
                category_sample = category_df.sample(
                    n = analysis_params['samples'], random_state = 1
                    )
            else:
                category_sample = category_df.copy()
        
            # ---------- add data ----------
            axs[row,col].scatter(category_sample[analysis_params['x_var']],
                                 category_sample[analysis_params['y_var']], 
                                 s = plot_params['m_size'], c = c[j], 
                                 edgecolors = ec[j], 
                                 alpha = plot_params['m_alpha'], 
                                 linewidths = plot_params['linewidth']
                                 )
    
    # ----- annotate and set subplot properties -----
    for j in range(0,num_categories):
        row = int(j/(subplot_dims[1]))
        col = j - row*(subplot_dims[1])
        # plot field average vs x_var
        axs[row,col].plot(field_avg_x, field_avg_var, c = 'k',
                          linewidth = plot_params['linewidth'], 
                          linestyle = plot_params['linestyle'])
        
        # set axes parameters
        axs[row,col].set_ylim(plot_params['ylims'])
        axs[row,col].set_xlim(plot_params['xlims'])
        axs[row,col].set_ylabel(analysis_params['y_var'], 
                                fontsize = plot_params['fontsize'])
        axs[row,col].set_xlabel(analysis_params['x_var'], 
                                fontsize = plot_params['fontsize'])
        axs[row,col].tick_params(labelsize = plot_params['fontsize'])
        axs[row,col].grid(True, alpha = plot_params['grid_alpha'], zorder = 0) 
        if plot_params['log_x']:
            axs[row,col].set_xscale('log')
        else:
            axs[row,col].set_xscale('linear')
        
        # ---- add title to figures ----
        if j == num_categories-1:
            axs[row,col].set_title(analysis_params['cat_var']+ '_band: '+
                                   '>' + str(round(category_ranges[j],1)),
                                   fontsize = plot_params['fontsize']
                                   )
        else:       
            axs[row,col].set_title(analysis_params['cat_var'] + '_band: ' + 
                                   str(round(category_ranges[j],2)) + ':' + 
                                   str(round(category_ranges[j+1],2)), 
                                   fontsize = plot_params['fontsize']
                                   )
            
        # annotate showing mask frame with vertical dashed line
        _, max_ylim = plt.ylim()
        if analysis_params['x_var'] == 'time':
            axs[row,col].axvline(time_mapping[analysis_params['mask_frame']], 
                                 linestyle = plot_params['annot_linestyle'], 
                                 linewidth = plot_params['linewidth'],
                                 marker = '')
            axs[row,col].text(time_mapping[analysis_params['mask_frame']]*1.1, 
                              max_ylim*0.8, 
                              'Mask frame: {:.0f}'.format(time_mapping[analysis_params['mask_frame']])+ 
                              ' s', fontsize = plot_params['fontsize'])
        else:
            # add line showning mean of field at each frame
            axs[row,col].axvline(analysis_params['mask_frame'], 
                                 linestyle = plot_params['annot_linestyle'], 
                                 linewidth = plot_params['linewidth'], 
                                 marker = '')
            axs[row,col].text(analysis_params['mask_frame']*1.1, max_ylim*0.8,
                          'Mask frame: {:.0f}'.format(mask_frame),
                          fontsize = plot_params['fontsize']) 
    plt.tight_layout()        
    plt.show()
    
    
def overlay_pts_on_sample(plot_params, mask_frame, num_categories, 
                          category_indices, category_ranges, 
                          load_multiple_frames, dir_results, 
                          img_scale, c, data_df = None):
    '''Overlay location of cluster points on undeformed sample
    
    Args: 
        plot_params (dict): dictionary of parameters to customize plot
        mask_frame (int): frame used to mask sample for clustering
        num_categories (int): number of categories used to cluster points
        category_indices (dict): contains index series of points corresponding
            to each category
        category_ranges (array): bounds on ranges used to cluster sample
        load_multiple_frames (bool): flag to process in batch or frame-by-frame
        dir_results (str): dic results dictionary
        img_scale (float): mm/pixel scale for images
        c (array): list of possible marker colours
        data_df (dataframe, optional): pre-loaded results from all frames in
            one data structure
            
    Returns:
        Figure

    '''
    # ------------------------------------------------------------------------
    # ----- create figure -----
    # ------------------------------------------------------------------------
    f = plt.figure(figsize = plot_params['figsize'])
    ax = f.add_subplot(1,1,1)
    # ---------- load data for reference and mask frames ----------
    if load_multiple_frames: 
            frame_df = data_df[data_df['frame'] == mask_frame]
            reference_df = data_df[data_df['frame'] == 1]
    else:
        frame_df = return_frame_dataframe(mask_frame, dir_results)
        reference_df = return_frame_dataframe(1, dir_results)
        
    # plot reference points
    ax.scatter(reference_df[['x_pix']]*img_scale, 
            reference_df[['y_pix']]*img_scale, 
            s = plot_params['m_size'], c = plot_params['ref_c'], 
            edgecolors =  plot_params['ref_c'], 
            alpha = plot_params['ref_alpha'],
            linewidths = plot_params['linewidth'], 
            zorder = 0, label = 'Other'
            )
    # ---------- add data ----------
    for i in range(0,num_categories):
        category_df = frame_df[frame_df.index.isin(category_indices[i].values)]
    
        ax.scatter(
            category_df[['x_pix']]*img_scale, category_df[['y_pix']]*img_scale, 
            s = plot_params['m_size'], c = c[i], edgecolors = c[i], 
            alpha = plot_params['cluster_alpha'], 
            linewidths = plot_params['linewidth'], zorder = i,
            label = 'Cluster: ' + str(i)
            )
    
    # ----- customize plot - add grid, labels, and legend -----
    ax.grid(True, alpha = plot_params['grid_alpha'], zorder = 0)
    ax.set_xlabel(plot_params['xlabel'], fontsize = plot_params['fontsize'])
    ax.set_ylabel(plot_params['ylabel'], fontsize = plot_params['fontsize'])
    ax.set_xlim(plot_params['xlims'])
    ax.set_ylim(plot_params['ylims'])
    ax.tick_params(labelsize = plot_params['fontsize'])
    if plot_params['axes_scaled']:
        ax.axis('scaled')
    
    # ----- customize legend -----
    box = ax.get_position()
    ax.set_position(
        [box.x0, box.y0 + box.height * 0.25,
         box.width, box.height * 0.5]
        )
    legend = ax.legend(
        fontsize = 3, loc='upper center', ncol=3, 
        bbox_to_anchor=(0.5, 1.13), fancybox = False
        )
    legend.get_frame().set_linewidth(0.5)
    
    for handle in legend.legendHandles:
        handle.set_sizes([plot_params['m_legend_size']])
    
    if plot_params['tight_layout']:
        plt.tight_layout()
        
    plt.show()
    
def plot_compressibility_check_clusters(analysis_params, plot_params, 
                                       num_categories, category_indices, 
                                       plot_frame_range, load_multiple_frames,
                                       dir_results, c, ec, data_df = None):
    '''Plot strain tensor components to check compressibility behaviour
    
    Args:
        analysis_params (dict): variables used for plot on x,y and categories
        plot_params (dict): dictionary of parameters to customize plot
        num_categories (int): number of categories used to cluster points
        category_indices (dict): contains index series of points corresponding
            to each category
        plot_frame_range (array): min and max frame numbers to plot
        load_multiple_frames (bool): flag to process in batch or frame-by-frame
        dir_results (str): dic results dictionary
        img_scale (float): mm/pixel scale for images
        c (array): list of possible marker colours
        ec (array): list of possible marker edge colours
        data_df (dataframe, optional): pre-loaded results from all frames in
            one data structure
            
    Returns:
        Figure

    '''
    # ------------------------------------------------------------------------
    # ----- create figure -----
    # ------------------------------------------------------------------------
    f = plt.figure(figsize = plot_params['figsize'])
    ax = f.add_subplot(1,1,1)
    
    # compare against incompressible relationship    
    ax.plot(
        analysis_params['x_fit'], analysis_params['y_fit_1'], 
        linewidth = plot_params['linewidth'], 
        linestyle = plot_params['linestyle1'], c = 'k',  
        label = plot_params['y_fit_1_label'])
    ax.plot(
        analysis_params['x_fit'], analysis_params['y_fit_2'], 
        linewidth = plot_params['linewidth'],
        linestyle = plot_params['linestyle2'], c = 'k', 
        label = plot_params['y_fit_2_label'])
    
    for i in range(plot_frame_range[0],plot_frame_range[1]+1):
    # ---------- load data for current frame ----------
        if load_multiple_frames: 
            frame_df = data_df[data_df['frame'] == i]
        else:
            frame_df = return_frame_dataframe(i, dir_results)
            
        for j in range(0,num_categories):
            
            category_df = frame_df[frame_df.index.isin(category_indices[j].values)]
            
            if category_df.shape[0] > analysis_params['samples']:
                category_sample = category_df.sample(
                    n = analysis_params['samples'], random_state = 1
                    )
            else:
                category_sample = category_df.copy()
        
            # ---------- add data ----------
            ax.scatter(category_sample[analysis_params['x_var']],
                                 category_sample[analysis_params['y_var']], 
                                 s = plot_params['m_size'], c = c[j], 
                                 edgecolors = ec[j], 
                                 alpha = plot_params['m_alpha'], 
                                 linewidths = plot_params['linewidth'],
                                 label = 'Cluster: ' + str(j)
                                 )
        if i == plot_frame_range[0]:
            # add legend on first pass
            legend = ax.legend(loc='upper right', 
                               fontsize = plot_params['legend_fontsize'])
            legend.get_frame().set_linewidth(plot_params['linewidth'])
    
    # set axes parameters
    ax.set_ylim(plot_params['ylims'])
    ax.set_xlim(plot_params['xlims'])
    ax.set_ylabel(analysis_params['y_var'], 
                            fontsize = plot_params['fontsize'])
    ax.set_xlabel(analysis_params['x_var'], 
                            fontsize = plot_params['fontsize'])
    ax.tick_params(labelsize = plot_params['fontsize'])
    ax.grid(True, alpha = plot_params['grid_alpha'], zorder = 0)

    plt.show()