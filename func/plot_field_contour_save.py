# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 14:23:32 2021

@author: jcv
"""
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_field_contour_save(xx, yy, zz, plot_params, i):
    '''Generate contour plot of field response for one variable at a specified 
    frame number
    
    Args: 
        xx (array/list): x coordinates
        yy (array/list): y coordinates
        zz (array/list): magnitude of variable at each coordinate
        plot_params (dict): dictionary of parameters to customize plot
        i (int): frame number
            
    Returns:
        Figure
    '''
    # toggle interactive mode
    if not plot_params['show_fig']:
        plt.ioff()
    
    # plot map    
    f = plt.figure(figsize = plot_params['figsize'])
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
    ax.grid(True, alpha = plot_params['grid_alpha'], zorder = -1)
    ax.set_title('Frame: '+ str(i))
    cbar = f.colorbar(cf)
    cbar.ax.set_ylabel(plot_params['var_name'])

    # show grid but hide labels
    if plot_params['hide_labels']:
        ax.xaxis.set_ticklabels([])
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticklabels([])
        ax.yaxis.set_ticks_position('none')    
    else:
        ax.set_xlabel(plot_params['xlabel'])
        ax.set_ylabel(plot_params['ylabel'])
    
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
