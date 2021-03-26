# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 12:16:39 2021

@author: jcv
"""
# load in pickle file and plot variables of interest

import os
import sys
import csv
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.font_manager as font_manager
from mpl_toolkits.axes_grid1 import make_axes_locatable

#%% Define functions
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
        f.savefig(fpath, dpi=pp['dpi'], facecolor='w',
                  edgecolor='w', pad_inches = 0.1
                  )

#%% ---- MAIN SCRIPT ----
# ----- configure directories -----
# root directory
dir_root = 'Z:/Experiments/lce_tension'
# extensions to access sub-directories
batch_ext = 'lcei_001'
mts_ext = 'mts_data'
sample_ext = '007_t01_r00'
gom_ext = 'gom_results'

# define full paths to mts and gom data
dir_gom_results = os.path.join(dir_root,batch_ext,sample_ext,gom_ext)

# check if figures folder exists, if not, make directory
if not os.path.exists(os.path.join(dir_gom_results,'figures')):
    os.makedirs(os.path.join(dir_gom_results,'figures'))

# define directory where figures to be saved    
dir_figs_root = os.path.join(dir_gom_results,'figures')

if not os.path.exists(os.path.join(dir_figs_root,'disp_fields')):
    os.makedirs(os.path.join(dir_figs_root,'disp_fields'))
    
if not os.path.exists(os.path.join(dir_figs_root,'strain_fields')):
    os.makedirs(os.path.join(dir_figs_root,'strain_fields'))
    
if not os.path.exists(os.path.join(dir_figs_root,'rotation_fields')):
    os.makedirs(os.path.join(dir_figs_root,'rotation_fields'))

dir_disp_folder = os.path.join(dir_figs_root,'disp_fields')
dir_strain_folder = os.path.join(dir_figs_root,'strain_fields')
dir_rotation_folder = os.path.join(dir_figs_root,'rotation_fields')

# ----- define constants -----
spec_id = batch_ext+'_'+sample_ext # full specimen id
Ny, Nx = 2448, 2048 # pixel resolution in x, y axis
img_scale = 0.0106 # mm/pix
t = 1.6 # thickness of sample [mm]
cmap_name = 'lajolla' # custom colormap stored in mpl_styles
cbar_levels = 25 # colorbar levels

# load in colormap and define plot style
cm_data = np.loadtxt('Z:/Python/mpl_styles/'+cmap_name+'.txt')
custom_map = LinearSegmentedColormap.from_list('custom', cm_data)

plt.style.use('Z:/Python/mpl_styles/stg_plot_style_1.mplstyle')

# find files ending with .pkl
files_pkl = [f for f in os.listdir(dir_gom_results) if f.endswith('.pkl')]

#%%
# load in data
frame_count = 0
for i in range(0,25):
    print('Processing frame: '+str(i))
    save_filename = 'results_df_frame_'+str(i)+'.pkl'
    frame_df = pd.read_pickle(os.path.join(dir_gom_results,save_filename))
        
    # add time stamp to frame to allow for sorting later
    frame_df['frame'] = i*np.ones((frame_df.shape[0],))
    
    if frame_count == 0:
        # create empty data frame to store all values from each frame
        all_frames_df = pd.DataFrame(columns = frame_df.columns)
    
    all_frames_df = pd.concat(
        [all_frames_df, frame_df], 
        axis = 0, join = 'outer'
        )
    
    frame_count += 1

# add columns with scaled coordinates
all_frames_df['x_mm'] = all_frames_df['x_pix']*img_scale + all_frames_df['ux']
all_frames_df['y_mm'] = all_frames_df['y_pix']*img_scale + all_frames_df['uy']

#%% 
# create dictionary of plot parameters - pass to function
plot_params = {'figsize': (3,1), 'xlabel': 'x (mm)', 'ylabel': 'y (mm)', 
               's': 0.5, 'xmin': 0, 'xmax': 28,
               'ymin': 7, 'ymax': 13,
               'dpi': 200, 'cmap': custom_map,
               'tight_layout': True, 'hide_labels': False, 'show_fig': False,
               'save_fig': True
               }   

for i in range(25,47):
    print('Plotting fields for frame: '+str(i))
    for j in ['ux','uy','Exx','Eyy','Exy','R']:
        
        # filter data to plot
        xx = np.array(all_frames_df[all_frames_df['frame'] == i][['x_mm']])
        yy = np.array(all_frames_df[all_frames_df['frame'] == i][['y_mm']])
        zz = np.array(all_frames_df[all_frames_df['frame'] == i][[j]])
        
        # set limits for plotting based on last frame pd.describe()
        if j == 'Exx':
            vmin = 0
            vmax = 6.5
            dir_save_figs = dir_strain_folder
        elif j == 'Eyy':
            vmin = -0.4
            vmax = 0.4
            dir_save_figs = dir_strain_folder
        elif j == 'Exy':
            vmin = -1
            vmax = 1
            dir_save_figs = dir_strain_folder
        elif j == 'ux':
            vmin = 0
            vmax = 22
            dir_save_figs = dir_disp_folder
        elif j == 'uy':
            vmin = -1.2
            vmax = 1.2
            dir_save_figs = dir_disp_folder
        else:
            vmin = 0
            vmax = 12
            dir_save_figs = dir_rotation_folder
        
        # assign additional key:value pairs to plot params dictionary
        plot_params['vmin'] = vmin
        plot_params['vmax'] = vmax
        plot_params['var_name'] = j
        
        # define full path of image
        fpath = dir_save_figs+'/'+spec_id+'_'+j+'_'+str(i)+'.tiff'
        
        plot_params['fpath'] = fpath
        
        # pass file to function that plots fields to file - lower res to save time initially
        plot_field_contour_save(xx, yy, zz, plot_params)

#%%




