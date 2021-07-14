# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 14:23:32 2021

@author: jcv
"""
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_field_contour_save(xx, yy, zz, pp):
    # toggle interactive mode
    if not pp['show_fig']:
        plt.ioff()
    
    # plot map    
    f = plt.figure(figsize = pp['figsize'])
    ax = f.add_subplot(1,1,1)
    cf = ax.scatter(xx, yy, c = zz, s = pp['s'],  
                        vmin = pp['vmin'], vmax = pp['vmax'], 
                        cmap = pp['cmap']
                         )
    
    # set axes features/limits
    ax.set_xlim([pp['xmin'], pp['xmax']])
    ax.set_ylim([pp['ymin'], pp['ymax']])
    ax.grid(True, alpha = 0.4, zorder = -1)
    cbar = f.colorbar(cf)
    cbar.ax.set_ylabel(pp['var_name'])

    # show grid but hide labels
    if pp['hide_labels']:
        ax.xaxis.set_ticklabels([])
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticklabels([])
        ax.yaxis.set_ticks_position('none')    
    else:
        ax.set_xlabel(pp['xlabel'])
        ax.set_ylabel(pp['ylabel'])
    
    if pp['tight_layout']:
        plt.tight_layout()  
    
    # toggle interactive mode
    if pp['show_fig']:
        plt.show()

    # save figure
    if pp['save_fig']:
        f.savefig(pp['fpath'], dpi=pp['dpi'], facecolor='w',
                  edgecolor='w', pad_inches = 0.1
                  )