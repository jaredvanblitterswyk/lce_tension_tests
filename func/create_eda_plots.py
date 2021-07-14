# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 16:46:54 2021

@author: jcv
"""
import matplotlib.pyplot as plt 

def create_simple_scatter(x, y, plot_params, c, ec):
    f = plt.figure(figsize = plot_params['figsize'])
    ax = f.add_subplot(1,1,1)
    ax.scatter(x = x, y = y, s = plot_params['m_size'], 
               c = c[0], edgecolors = ec[0], 
               linewidths = plot_params['linewidth'])
    ax.set_xlabel(plot_params['xlabel'])
    ax.set_ylabel(plot_params['ylabel'])
    ax.grid(zorder=0)
    if plot_params['log_x']:
        ax.set_xscale('log')
    
    if plot_params['tight_layout']:
        plt.tight_layout()
    plt.show()
    
def plot_boxplot_vs_frame(data, frame_labels, ylabel):
    # plot exx
    f = plt.figure(figsize = (6,3))
    ax = f.add_subplot(1,1,1)
    ax.boxplot(data)#, boxprops = boxprops, flierprops = flierprops, 
               #whiskerprops = boxprops, medianprops = medianprops, 
               #capprops = boxprops)
    ax.set_xticklabels(frame_labels)
    ax.set_xlabel('Frame')
    ax.set_ylabel(ylabel)
    ax.grid(zorder=0)
    plt.tight_layout()