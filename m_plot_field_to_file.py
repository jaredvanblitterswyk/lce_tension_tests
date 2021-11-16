# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 12:16:39 2021

@author: jcv
"""
# load in pickle file and plot variables of interest

import os
import sys
sys.path.append('Z:/Python/tension_test_processing')
sys.path.append(os.path.join(sys.path[-1],'func'))
import csv
import pandas as pd
import numpy as np
import json
import math as m
import matplotlib as mpl
from func.plot_field_contour_save import *
import processing_params as udp
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

#%% ---- INITIALIZE DIRECTORIES ----
# check if figures folder exists, if not, make directory
if not os.path.exists(os.path.join(udp.dir_figs_root)):
    os.makedirs(os.path.join(udp.dir_figs_root))

if not os.path.exists(os.path.join(udp.dir_disp_fields)):
    os.makedirs(os.path.join(udp.dir_disp_fields))
    
if not os.path.exists(os.path.join(udp.dir_nu_fields)):
    os.makedirs(os.path.join(udp.dir_nu_fields))
    
if not os.path.exists(os.path.join(udp.dir_strain_fields)):
    os.makedirs(os.path.join(udp.dir_strain_fields))
    
if not os.path.exists(os.path.join(udp.dir_rotation_fields)):
    os.makedirs(os.path.join(udp.dir_rotation_fields))

#%% ---- GENERATE DIC MAPS FOR SPECIFIED VARIABLE -----
# load plot style
plt.style.use(udp.dir_plt_style)

# define full specimen id for figure filenames
spec_id = udp.batch_ext + '_' + udp.sample_ext

# find files ending with .pkl
files_pkl = [f for f in os.listdir(udp.dir_gom_results) if f.endswith('.pkl')]

# load general plot params from processing file
plot_params = udp.plt_params_fields_general

# load in data
for i in range(udp.plt_map_frame_range[0], udp.plt_map_frame_range[1]):
    print('Processing frame: '+str(i))
    save_filename = 'results_df_frame_'+"{:02d}".format(i)+'.pkl'
    frame_df = pd.read_pickle(os.path.join(udp.dir_gom_results, save_filename))
           
    # add columns with scaled coordinates
    frame_df['x_mm'] = frame_df['x_pix']*udp.img_scale + frame_df['ux']
    frame_df['y_mm'] = frame_df['y_pix']*udp.img_scale + frame_df['uy']
    
    # create in-plane Poisson's ratio feature
    try: 
        if udp.orientation == 'vertical':
            frame_df['nu'] = -1*frame_df['Exx']/frame_df['Eyy']
        elif udp.orientation == 'horizontal':
            frame_df['nu'] = -1*frame_df['Eyy']/frame_df['Exx']
    except:
        print('Specimen orientation not recognized/specified.')
    
    frame_df['nu'] = frame_df['nu'].apply(lambda x: x if x >= 0 else 0)

    plt.close('all')
    print('Plotting fields for frame: '+str(i))
    for j in udp.vars_to_plot_map:
        
        # filter data to plot
        xx = np.array(frame_df[['x_mm']])
        yy = np.array(frame_df[['y_mm']])
        zz = np.array(frame_df[[j]])
        
        # assign variable-specific key:value pairs to plot params dictionary
        plot_params['vmin'] = udp.plt_params_var_clims[j]['vlims'][0]
        plot_params['vmax'] = udp.plt_params_var_clims[j]['vlims'][1]
        plot_params['var_name'] = j
        
        # define full path of image
        fpath = udp.plt_params_var_clims[j]['dir_save_figs'] + '/' + spec_id \
            + '_' + j + '_'+str(i) + udp.map_img_type_save
        
        plot_params['fpath'] = fpath
        
        # pass file to function that plots fields to file - lower res to save time initially
        plot_field_contour_save(xx, yy, zz, plot_params, 'Frame: '+ str(i))

#%% ----- WRITE PROCESSING AND PLOT PARAMETERS TO FILE -----
# define output path
output_filename = spec_id + '_plot_config.json'
out_path = os.path.join(udp.dir_figs_root, output_filename)

# store output in dictionary
output = []
output.append(
    {
     'Plot_parameters': '', 
     'img_scale': udp.img_scale, 
     'thickness': udp.thickness,
     'plot_style': 'Z:/Python/mpl_styles/' + udp.cmap_name+'.txt'
     }
    )

# remove colourmap object
plot_params.pop('cmap', None)
output.append(plot_params)
output.append(udp.plt_params_var_clims)

with open(out_path, 'w',  encoding='utf-8') as f:
    json.dump(output, f, indent=2)