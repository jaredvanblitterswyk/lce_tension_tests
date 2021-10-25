# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 14:23:32 2021

@author: jcv
"""
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_field_contour_manuscript(xx, yy, zz, plot_params, title = None):
    '''Generate contour plot of field response for one variable at a specified 
    frame number
    
    Args: 
        xx (array/list): x coordinates
        yy (array/list): y coordinates
        zz (array/list): magnitude of variable at each coordinate
        plot_params (dict): dictionary of parameters to customize plot
        title (string): figure title (optional)
            
    Returns:
        Figure
    '''
    v1 = np.linspace(plot_params['vmin'], plot_params['vmax'], 7, endpoint=True)   
    
    # toggle interactive mode
    if not plot_params['show_fig']:
        plt.ioff()
    
    # plot map    
    f = plt.figure(figsize = plot_params['figsize'])
    f.subplots_adjust(left = plot_params['bbox'][0],
                      right = plot_params['bbox'][1],
                      top = plot_params['bbox'][2],
                      bottom = plot_params['bbox'][3])
    ax = f.add_subplot(1,1,1)
    cf = ax.scatter(xx, yy, c = zz, 
                    s = plot_params['m_size'],  
                    vmin = plot_params['vmin'], 
                    vmax = plot_params['vmax'], 
                    cmap = plot_params['cmap']
                    )
    
    # set axes features/limits
    ax.set_xlim(plot_params['xlims'])
    ax.set_ylim(plot_params['ylims'])
    ax.grid(True, alpha = plot_params['grid_alpha'], 
            linewidth = plot_params['grid_lw'], zorder = -1)
    if title:
        ax.set_title(title)
    if plot_params['cbar']:
        cbar = f.colorbar(cf, ticks=v1)
        cbar.ax.set_ylabel(plot_params['var_name'])
        cbar.ax.set_yticklabels(["{:4.2f}".format(i) for i in v1])

    # show grid but hide labels
    if plot_params['hide_labels']:
        ax.xaxis.set_ticklabels([])
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticklabels([])
        ax.yaxis.set_ticks_position('none')    
    else:
        ax.set_xlabel(plot_params['xlabel'], fontsize = plot_params['fontsize'])
        ax.set_ylabel(plot_params['ylabel'], fontsize = plot_params['fontsize'])
        ax.tick_params(axis='both', which='major', labelsize = plot_params['fontsize'])
        
    for axis in ['left', 'right', 'top', 'bottom']:
        ax.spines[axis].set_linewidth(plot_params['spine_lw'])
    
    if plot_params['tight_layout']:
        plt.tight_layout()  
    
    # toggle interactive mode
    if plot_params['show_fig']:
        plt.show()

    # save figure
    if plot_params['save_fig']:
        try:
            assert plot_params['fpath']
            f.savefig(plot_params['fpath'], dpi=plot_params['dpi'], 
                      facecolor='w', edgecolor='w', pad_inches = 0.1
                      )
        
        except:
            print('No file path provided, figure not saved.')
            pass
