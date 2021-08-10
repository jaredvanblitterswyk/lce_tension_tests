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

def create_simple_scatter(x, y, plot_params, plot_frame_range, ax):
    '''Generate boxplots for one variable as a function of frame number
    
    Args: 
        x (array/list): x values to plot
        y (array/list): y values to plot
        plot_params (dict): dictionary of parameters to customize plot
        plot_frame_range (array): min and max frame numbers to plot
            
    Returns:
        Figure
    '''
            
    # ---------- add data ----------   
    ax.scatter(x, y, s = plot_params['m_size'], marker = plot_params['marker'],
               c = plot_params['c'][0], edgecolors = plot_params['ec'][0], 
               linewidths = plot_params['linewidth'], 
               label = plot_params['label'])
    ax.set_xlabel(plot_params['xlabel'], fontsize = plot_params['fontsize'])
    ax.set_ylabel(plot_params['ylabel'], fontsize = plot_params['fontsize'])
    ax.grid(True, alpha = plot_params['grid_alpha'], zorder=0)
    ax.tick_params(labelsize = plot_params['fontsize'])
    legend = ax.legend(loc=plot_params['legend_loc'], 
                       fontsize = plot_params['legend_fontsize'])
    legend.get_frame().set_linewidth(plot_params['linewidth'])
    if plot_params['log_x']:
        ax.set_xscale('log')
        
    
def boxplot_vs_frame(frame_df, analysis_params, plot_params, 
                              plot_frame_range, i, ax):
    '''Generate boxplots for one variable as a function of frame number
    
    Args: 
        frame_df (dataframe): results from current frame
        plot_var (str): variable to plot
        plot_params (dict): dictionary of parameters to customize plot
        plot_frame_range (array): min and max frame numbers to plot
        i (int): current frame
        ax (object): handle of current axes
            
    Returns:
        Figure

    '''
    # genenerate frame labels
    frame_labels = [f for f in range(plot_frame_range[0],plot_frame_range[1]+1)]
    
    # ---------- add data ----------        
    ax.boxplot(frame_df[analysis_params['plot_var']].values, positions = [i], 
               showfliers = plot_params['showfliers'])
    ax.set_xticklabels(frame_labels)
    ax.set_xlabel(plot_params['xlabel'], fontsize = plot_params['fontsize'])
    ax.set_ylabel(plot_params['ylabel'], fontsize = plot_params['fontsize'])
    ax.set_xlim(plot_params['xlims'])
    ax.tick_params(labelsize = plot_params['fontsize'])
    ax.grid(True, alpha = plot_params['grid_alpha'], zorder = 0)

    
def histogram_vs_frame(frame_df, analysis_params, plot_params, 
                       plot_frame_range, plot_num, i, ax):
    '''Generate histogram for mulitple frames in separate subplots
    
    Args:
        frame_df (dataframe): results from current frame
        subplot_dims (array): number of subplots in rows and columns
        plot_var (str): variable to plot
        plot_params (dict): dictionary of parameters to customize plot
        plot_frame_range (array): min and max frame numbers to plot
        plot_num (int): index of current plot
        i (int): current frame
        ax (object): handle of current axes
            
    Returns:
        Figure

    '''
    
    # --------- compute axis indices ----------
    row = int(plot_num/(plot_params['subplot_dims'][1]))
    col = plot_num - row*(plot_params['subplot_dims'][1])
    
    # ---------- generate subplot ----------
    ax[row,col].hist(frame_df[analysis_params['plot_var']], 
                     edgecolor = plot_params['ec'][0], 
                     color = plot_params['c'][0], 
                     bins = plot_params['n_bins'], 
                     linewidth = plot_params['linewidth'])
    ax[row,col].set_title('Frame: '+ str(i), fontsize = plot_params['fontsize'])
    ax[row,col].set_xlabel(analysis_params['plot_var'], 
                           fontsize = plot_params['fontsize'])
    ax[row,col].grid(True, alpha = plot_params['grid_alpha'], zorder = 0)
    ax[row,col].set_xlim(plot_params['xlims'])
    ax[row,col].tick_params(labelsize = plot_params['fontsize'])
    
    # extract ylims of plot for annotations
    _, max_ylim = ax[row,col].get_ylim()
    
    # add line showning mean of field at each frame
    avg_strain = frame_df[analysis_params['plot_var']].mean()
    ax[row,col].axvline(avg_strain,
        color='k', linestyle=plot_params['linestyle2'], 
        linewidth = plot_params['linewidth2'], marker = ''
        )
    ax[row,col].text(
        avg_strain*1.1, max_ylim*0.8, 
        'Mean: {:.2f}'.format(avg_strain), 
        fontsize = plot_params['fontsize2']
        ) 
    
def var_clusters_vs_time_subplots(frame_df, analysis_params, plot_params, 
                                  plot_frame_range, time_mapping, 
                                  field_avg_var, field_avg_x, i, ax):
    '''Generate scatter plots of values from points on the sample belonging to
    categories defined based on an input variable and mask frame
    
    Args: 
        frame_df (dataframe): results from current frame
        subplot_dims (array): number of subplots in rows and columns
        analysis_params (dict): variables used for plot on x,y and categories
        plot_params (dict): dictionary of parameters to customize plot
        plot_frame_range (array): min and max frame numbers to plot
        time_mapping (dict): map frame number to test time
        field_avg_var (array): field average of y variable for all frames
        field_avg_x (array): field average of x variable for all frames
        i (int): current frame
        ax (object): handle of current axes        
            
    Returns:
        Figure

    '''

    # ----- add data to subplots -----  
    for j in range(0, analysis_params['num_categories']):
        row = int(j/(plot_params['subplot_dims'][1]))
        col = j - row*(plot_params['subplot_dims'][1])
        
        category_df = frame_df[frame_df.index.isin(
                    analysis_params['category_indices'][j].values
                    )]
        
        if category_df.shape[0] > analysis_params['samples']:
            category_sample = category_df.sample(
                n = analysis_params['samples'], random_state = 1
                )
        else:
            category_sample = category_df.copy()
    
        ax[row,col].scatter(category_sample[analysis_params['x_var']],
                             category_sample[analysis_params['y_var']], 
                             s = plot_params['m_size'], 
                             c = plot_params['c'][j], 
                             edgecolors = plot_params['ec'][j], 
                             alpha = plot_params['m_alpha'], 
                             linewidths = plot_params['linewidth']
                             )
    
    # ----- annotate and set subplot properties (last frame only) -----
    if i == plot_frame_range[1]:
        for j in range(0,analysis_params['num_categories']):
            row = int(j/(plot_params['subplot_dims'][1]))
            col = j - row*(plot_params['subplot_dims'][1])
            # plot field average vs x_var
            ax[row,col].plot(field_avg_x, field_avg_var, c = 'k',
                              linewidth = plot_params['linewidth'], 
                              linestyle = plot_params['linestyle'])
            
            # set axes parameters
            ax[row,col].set_ylim(plot_params['ylims'])
            ax[row,col].set_xlim(plot_params['xlims'])
            ax[row,col].set_ylabel(analysis_params['y_var'], 
                                    fontsize = plot_params['fontsize'])
            ax[row,col].set_xlabel(analysis_params['x_var'], 
                                    fontsize = plot_params['fontsize'])
            ax[row,col].tick_params(labelsize = plot_params['fontsize'])
            ax[row,col].grid(True, alpha = plot_params['grid_alpha'], zorder = 0) 
            if plot_params['log_x']:
                ax[row,col].set_xscale('log')
            else:
                ax[row,col].set_xscale('linear')
            
            # ---- add title to figures ----
            if j == analysis_params['num_categories']-1:
                ax[row,col].set_title(
                    analysis_params['cat_var']+ '_band: '+
                    '>' + str(round(analysis_params['category_ranges'][j],1)),
                    fontsize = plot_params['fontsize']
                    )
            else:       
                ax[row,col].set_title(
                    analysis_params['cat_var'] + '_band: ' + 
                    str(round(analysis_params['category_ranges'][j],2)) + ':' + 
                    str(round(analysis_params['category_ranges'][j+1],2)), 
                    fontsize = plot_params['fontsize']
                    )
                
            # annotate showing mask frame with vertical dashed line
            _, max_ylim = ax[row,col].get_ylim()
            if analysis_params['x_var'] == 'time':
                ax[row,col].axvline(time_mapping[analysis_params['mask_frame']], 
                                     linestyle = plot_params['linestyle2'], 
                                     linewidth = plot_params['linewidth2'],
                                     marker = '')
                ax[row,col].text(time_mapping[analysis_params['mask_frame']]*1.1, 
                                  max_ylim*0.8, 
                                  'Mask frame: {:.0f}'.format(
                                      time_mapping[analysis_params['mask_frame']]) + 
                                  ' s', fontsize = plot_params['fontsize2'])
            else:
                # add line showning mean of field at each frame
                ax[row,col].axvline(analysis_params['mask_frame'], 
                                     linestyle = plot_params['linestyle2'], 
                                     linewidth = plot_params['linewidth2'], 
                                     marker = '')
                ax[row,col].text(analysis_params['mask_frame']*1.1, max_ylim*0.8,
                              'Mask frame: {:.0f}'.format(mask_frame),
                              fontsize = plot_params['fontsize2']) 
    
    
def overlay_pts_on_sample(plot_params, reference_df, mask_frame_df, 
                          analysis_params, img_scale, ax):
    '''Overlay location of cluster points on undeformed sample
    
    Args: 
        plot_params (dict): dictionary of parameters to customize plot
        reference_df (dataframe): dataframe of first frame in series
        mask_frame_df (dataframe): dataframe for mask frame for clustering
        analysis_params (dict): dictionary of parameters used in analysis
        img_scale (float): mm/pixel scale for images
        ax (object): handle of current axes  
            
    Returns:
        Figure

    '''
        
    # plot reference points
    ax.scatter(reference_df[['x_pix']]*img_scale, 
            reference_df[['y_pix']]*img_scale, marker = plot_params['marker'],
            s = plot_params['m_size'], c = plot_params['ref_c'], 
            edgecolors =  plot_params['ref_c'], 
            alpha = plot_params['m_alpha'],
            linewidths = plot_params['linewidth'], 
            zorder = 0, label = 'Other'
            )
    # ---------- add cluster points ----------
    for i in range(0,analysis_params['num_categories']):
        category_df = mask_frame_df[mask_frame_df.index.isin(
            analysis_params['category_indices'][i].values
            )]
    
        ax.scatter(
            category_df[['x_pix']]*img_scale, category_df[['y_pix']]*img_scale,
            marker = plot_params['marker'], s = plot_params['m_size'], 
            c = plot_params['c'][i], edgecolors = plot_params['c'][i], 
            alpha = plot_params['m_alpha'], 
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
        fontsize = plot_params['legend_fontsize'], loc=plot_params['legend_loc'], 
        ncol= plot_params['legend_ncol'], bbox_to_anchor=(0.5, 1.15), 
        fancybox = plot_params['legend_fancybox']
        )
    legend.get_frame().set_linewidth(plot_params['legend_linewidth'])
    
    for handle in legend.legendHandles:
        handle.set_sizes([plot_params['legend_m_size']])
    
def compressibility_check_clusters(frame_df, analysis_params, plot_params,
                                       plot_frame_range, i, ax):
    '''Plot strain tensor components to check compressibility behaviour
    
    Args:
        frame_df (dataframe): results from current frame
        analysis_params (dict): variables used for plot on x,y and categories
        plot_params (dict): dictionary of parameters to customize plot
        plot_frame_range (array): min and max frame numbers to plot
        i (int): current frame
        ax (object): handle of current axes
            
    Returns:
        Figure

    '''
    
    # compare against incompressible relationship 
    if i == plot_frame_range[0]:
        ax.plot(
            analysis_params['x_fit'], analysis_params['y_fit_1'], 
            linewidth = plot_params['linewidth'], 
            linestyle = plot_params['linestyle'], c = 'k',  
            label = plot_params['y_fit_1_label'])
        ax.plot(
            analysis_params['x_fit'], analysis_params['y_fit_2'], 
            linewidth = plot_params['linewidth'],
            linestyle = plot_params['linestyle2'], c = 'k', 
            label = plot_params['y_fit_2_label'])
    
             
    for j in range(0,analysis_params['num_categories']):
        
        category_df = frame_df[frame_df.index.isin(analysis_params['category_indices'][j].values)]
        
        if category_df.shape[0] > analysis_params['samples']:
            category_sample = category_df.sample(
                n = analysis_params['samples'], random_state = 1
                )
        else:
            category_sample = category_df.copy()
    
        # ---------- add data ----------
        ax.scatter(category_sample[analysis_params['x_var']],
                   category_sample[analysis_params['y_var']],
                   marker = plot_params['marker'],
                   s = plot_params['m_size'], c = plot_params['c'][j], 
                   edgecolors = plot_params['ec'][j], 
                   alpha = plot_params['m_alpha'], 
                   linewidths = plot_params['linewidth'],
                   label = 'Cluster: ' + str(j)
                   )
        
    if i == plot_frame_range[0]:
        # add legend on first pass
        legend = ax.legend(loc=plot_params['legend_loc'], 
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
    
def var_vs_time_clusters_same_axis(frame_df, analysis_params, plot_params,
                               plot_frame_range, i, ax):
    '''Generate scatter plots of values from points on the sample belonging to
    categories defined based on an input variable and mask frame
    
    Args:
        frame_df (dataframe): results from current frame
        analysis_params (dict): variables used for plot on x,y and categories
        plot_params (dict): dictionary of parameters to customize plot
        plot_frame_range (array): min and max frame numbers to plot
        i (int): current frame
        ax (object): handle of current axes
            
    Returns:
        Figure

    '''
                   
    for j in range(0,analysis_params['num_categories']):

        category_df = frame_df[frame_df.index.isin(analysis_params['category_indices'][j].values)]
        
        if category_df.shape[0] > analysis_params['samples']:
            category_sample = category_df.sample(
                n = analysis_params['samples'], random_state = 1
                )
        else:
            category_sample = category_df.copy()
        
        # extract mean of all points in cluster
        x = category_sample.groupby(
            analysis_params['x_var'])[analysis_params['x_var']].mean()
        y = category_sample.groupby(
            analysis_params['x_var'])[analysis_params['y_var']].mean()
                
        # ---------- add data ----------
        ax.scatter(x, y, marker = plot_params['marker'],
                   s = plot_params['m_size'], c = plot_params['c'][j], 
                   edgecolors = plot_params['ec'][j], 
                   alpha = plot_params['m_alpha'], 
                   linewidths = plot_params['linewidth'],
                   label = plot_params['labels'][j]
                   )
            
    # add legend on first pass
    if i == plot_frame_range[0]:
        legend = ax.legend(loc=plot_params['legend_loc'], 
                           fontsize = plot_params['legend_fontsize'])
        legend.get_frame().set_linewidth(plot_params['linewidth'])
        for handle in legend.legendHandles:
            handle.set_sizes([plot_params['legend_m_size']])
        
        # set axes parameters
        ax.set_ylim(plot_params['ylims'])
        ax.set_xlim(plot_params['xlims'])
        ax.set_ylabel(analysis_params['y_var'], fontsize = plot_params['fontsize'])
        ax.set_xlabel(analysis_params['x_var'], fontsize = plot_params['fontsize'])
        ax.tick_params(labelsize = plot_params['fontsize'])
        ax.grid(True, alpha = plot_params['grid_alpha'], zorder = 0)
        if plot_params['log_x']:
                ax.set_xscale('log')
        else:
                ax.set_xscale('linear')  
    
def norm_stress_strain_rates_vs_time(analysis_params, plot_params, 
                                     plot_frame_range, i, ax):
    '''Plot normalized rate of change for two variables (intended to be Eyy 
    and stress), normalized by the value at peak load. The data is collected 
    and stored in category_series dictionary in a loop for ploting later
    
    Args: 
        analysis_params (dict): variables used for plot on x,y and categories
        plot_params (dict): dictionary of parameters to customize plot
        plot_frame_range (array): min and max frame numbers to plot
        i (int): current frame
        ax (object): handle of current axes
            
    Returns:
        Figure
        
    Notes: 
        Hard-coded summary dictionary to 2 keys for each variable representing
        clusters where category variable increases or decreases

    '''
    # ----- add data and set subplot properties (last frame only) -----
    for j in range(0,analysis_params['num_categories']):
        # extract y variable values at peak load (for normalization)
        Y1 = analysis_params['category_series']['y1_'+str(j)][analysis_params['peak_frame_index']]#)/analysis_params['dt'][analysis_params['peak_frame_index']]
        Y2 = analysis_params['category_series']['y2_'+str(j)][analysis_params['peak_frame_index']]#)/analysis_params['dt'][analysis_params['peak_frame_index']]
        
        #normalize data by value at peak load and compute relative change
        x_plot = analysis_params['x_series'][1:]
        y_plot = np.abs(np.diff(analysis_params['category_series']['y1_'+str(j)]/Y1))#analysis_params['dt'] / Y1 
        y_plot2 = np.abs(np.diff(analysis_params['category_series']['y2_'+str(j)]/Y2))#analysis_params['dt'] / Y2 
        
        # plot first variable
        ax.plot(x_plot, y_plot, linestyle = plot_params['linestyle'], 
                c = plot_params['c'][j],
                marker = plot_params['marker'],
                markeredgecolor = plot_params['ec'][j], 
                alpha = plot_params['m_alpha'], 
                linewidth = plot_params['linewidth'],
                label = plot_params['labels_y1'][j]
                )
        # plot second variable
        ax.plot(x_plot, y_plot2, linestyle = plot_params['linestyle2'], 
                c = plot_params['c'][j],
                marker = plot_params['marker2'],
                markeredgecolor = plot_params['ec'][j], 
                alpha = plot_params['m_alpha'], 
                linewidth = plot_params['linewidth'],
                label = plot_params['labels_y2'][j]
                )
        
    legend = ax.legend(loc=plot_params['legend_loc'], fontsize = plot_params['legend_fontsize'])
    legend.get_frame().set_linewidth(plot_params['linewidth'])
    
    # set axes parameters
    ax.set_ylim(plot_params['ylims'])
    ax.set_xlim(plot_params['xlims'])
    ax.set_ylabel(plot_params['ylabel'], fontsize = plot_params['fontsize'])
    ax.set_xlabel(plot_params['xlabel'], fontsize = plot_params['fontsize'])
    ax.tick_params(labelsize = plot_params['fontsize'])
    ax.grid(True, alpha = plot_params['grid_alpha'], zorder = 0)
    if plot_params['log_x']:
            ax.set_xscale('log')
    else:
            ax.set_xscale('linear')
    if plot_params['log_y']:
            ax.set_yscale('log')
    else:
            ax.set_yscale('linear')