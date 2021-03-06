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
import json
import math as m
import matplotlib as mpl
from func.plot_field_contour_save import *
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

#%% ---- MAIN SCRIPT ----
# ----- configure directories -----
# root directory
dir_root = 'Z:/Experiments/lce_tension'
# extensions to access sub-directories
batch_ext = 'lcei_003'
mts_ext = 'mts_data'
sample_ext = '001_t05_r00'
gom_ext = 'gom_results'
orientation = 'vertical'

# define full paths to mts and gom data
dir_gom_results = os.path.join(dir_root,batch_ext,sample_ext,gom_ext)

# check if figures folder exists, if not, make directory
if not os.path.exists(os.path.join(dir_gom_results,'figures')):
    os.makedirs(os.path.join(dir_gom_results,'figures'))

# define directory where figures to be saved    
dir_figs_root = os.path.join(dir_gom_results,'figures')

if not os.path.exists(os.path.join(dir_figs_root,'disp_fields')):
    os.makedirs(os.path.join(dir_figs_root,'disp_fields'))
    
if not os.path.exists(os.path.join(dir_figs_root,'nu_fields')):
    os.makedirs(os.path.join(dir_figs_root,'nu_fields'))
    
if not os.path.exists(os.path.join(dir_figs_root,'strain_fields')):
    os.makedirs(os.path.join(dir_figs_root,'strain_fields'))
    
if not os.path.exists(os.path.join(dir_figs_root,'rotation_fields')):
    os.makedirs(os.path.join(dir_figs_root,'rotation_fields'))

dir_disp_folder = os.path.join(dir_figs_root,'disp_fields')
dir_strain_folder = os.path.join(dir_figs_root,'strain_fields')
dir_rotation_folder = os.path.join(dir_figs_root,'rotation_fields')
dir_nu_folder = os.path.join(dir_figs_root,'nu_fields')

# ----- define constants -----
spec_id = batch_ext+'_'+sample_ext # full specimen id
Ny, Nx = 2048, 2448 # pixel resolution in x, y axis
img_scale = 0.02724 # mm/pix
t = 1.0 # thickness of sample [mm]
cmap_name = 'lapaz' # custom colormap stored in mpl_styles
cbar_levels = 25 # colorbar levels

# load in colormap and define plot style
cm_data = np.loadtxt('Z:/Python/mpl_styles/'+cmap_name+'.txt')
custom_map = LinearSegmentedColormap.from_list('custom', np.flipud(cm_data))

plt.style.use('Z:/Python/mpl_styles/stg_plot_style_1.mplstyle')

# find files ending with .pkl
files_pkl = [f for f in os.listdir(dir_gom_results) if f.endswith('.pkl')]

# create dictionary of plot parameters - pass to function
plot_params = {'figsize': (2,4),
               'xlabel': 'x (mm)', 
               'ylabel': 'y (mm)', 
               'm_size': 0.1, 
               'grid_alpha': 0.5,
               'dpi': 300, 'cmap': custom_map,
               'xlims': [24, 40],
               'ylims': [0, 45],#m.ceil(Ny*img_scale)],
               'tight_layout': True, 
               'hide_labels': False, 
               'show_fig': False,
               'save_fig': True
               }   
plot_var_specific = {'Exx': {
                'vlims': [-0.2, 0], 'dir_save_figs': dir_strain_folder
              },
              'Eyy': {
                  'vlims': [0.35, 0.6], 'dir_save_figs': dir_strain_folder
              },
              'Exy': {
                  'vlims': [-0.05, 0.05], 'dir_save_figs': dir_strain_folder
              },
              'ux': {
                  'vlims': [-1, 1], 'dir_save_figs': dir_disp_folder
              },
              'uy': {
                  'vlims': [0, 16], 'dir_save_figs': dir_disp_folder
              },
              'uz': {
                  'vlims': [0, 4], 'dir_save_figs': dir_disp_folder
              },
              'R': {
                  'vlims': [-4, 4], 'dir_save_figs': dir_rotation_folder
              },
              'nu': {
                  'vlims': [0, 0.5], 'dir_save_figs': dir_nu_folder
              }
              }

#%%
# load in data
frame_count = 0
for i in range(0,len(files_pkl)):
    print('Processing frame: '+str(i))
    save_filename = 'results_df_frame_'+"{:02d}".format(i)+'.pkl'
    frame_df = pd.read_pickle(os.path.join(dir_gom_results,save_filename))
           
    # add columns with scaled coordinates
    frame_df['x_mm'] = frame_df['x_pix']*img_scale + frame_df['ux']
    frame_df['y_mm'] = frame_df['y_pix']*img_scale + frame_df['uy']
    
    # create in-plane Poisson's ratio feature
    try: 
        if orientation == 'vertical':
            frame_df['nu'] = -1*frame_df['Exx']/frame_df['Eyy']
        elif orientation == 'horizontal':
            frame_df['nu'] = -1*frame_df['Eyy']/frame_df['Exx']
    except:
        print('Specimen orientation not recognized/specified.')
    
    frame_df['nu'] = frame_df['nu'].apply(lambda x: x if x >= 0 else 0)

    plt.close('all')
    print('Plotting fields for frame: '+str(i))
    for j in ['Eyy']:#['ux','uy','uz','Exx','Exy','Eyy','R', 'nu']:
        
        # filter data to plot
        xx = np.array(frame_df[['x_mm']])
        yy = np.array(frame_df[['y_mm']])
        zz = np.array(frame_df[[j]])
        
        # assign variable-specific key:value pairs to plot params dictionary
        plot_params['vmin'] = plot_var_specific[j]['vlims'][0]
        plot_params['vmax'] = plot_var_specific[j]['vlims'][1]
        plot_params['var_name'] = j
        
        # define full path of image
        fpath = plot_var_specific[j]['dir_save_figs']+'/'+spec_id+'_'+j+'_'+str(i)+'.tiff'
        
        plot_params['fpath'] = fpath
        
        # pass file to function that plots fields to file - lower res to save time initially
        plot_field_contour_save(xx, yy, zz, plot_params, i)

#%% 
# ----- write processing and plot parameters to file -----
# define output path
output_filename = batch_ext + '_'+ sample_ext + '_plot_config.json'
out_path = os.path.join(dir_figs_root,output_filename)

# store output in dictionary
output = []
output.append(
    {
     'Plot_parameters': '', 
     'img_scale': img_scale, 
     'thickness': t,
     'plot_style': 'Z:/Python/mpl_styles/'+cmap_name+'.txt'
     }
    )

# remove colourmap object
plot_params.pop('cmap', None)
output.append(plot_params)
output.append(plot_var_specific)

with open(out_path, 'w',  encoding='utf-8') as f:
    json.dump(output, f, indent=2)