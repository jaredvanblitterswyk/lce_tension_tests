# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 10:45:18 2021

@author: jcv
"""
import os
import sys
import csv
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

#%% ---- MAIN SCRIPT ----
# ----- configure directories -----
# root directory
dir_root = 'Z:/Experiments/lce_tension'
# extensions to access sub-directories
batch_ext = 'lcei_001'
mts_ext = 'mts_data'
sample_ext = '007_t01_r00'
gom_ext = 'gom_results'
cmap_name = 'lajolla' # custom colormap stored in mpl_styles

# define full paths to mts and gom data
dir_xsection = os.path.join(dir_root,batch_ext,sample_ext)
dir_mts = os.path.join(dir_root,batch_ext,mts_ext,batch_ext+'_'+sample_ext)
dir_gom_results = os.path.join(dir_root,batch_ext,sample_ext,gom_ext)

img_scale = 0.0106 # image scale (mm/pix)

for i in range(0,46):
    print('Adding frame: '+str(i))
    save_filename = 'results_df_frame_'+str(i)+'.pkl'
    frame_df = pd.read_pickle(os.path.join(dir_gom_results,save_filename))
    
    # add time stamp to frame to allow for sorting later
    frame_df['frame'] = i*np.ones((frame_df.shape[0],))
    
    if i == 0:
        # create empty data frame to store all values from each frame
        all_frames_df = pd.DataFrame(columns = frame_df.columns)
    
    all_frames_df = pd.concat([all_frames_df, frame_df], axis = 0, join = 'outer')
    
# filter data to track a specific point
x_track, y_track = 320, 875 # pixel coordinates of point to track

point_df = all_frames_df[(all_frames_df['x_pix'] == x_track) &(all_frames_df['y_pix'] == y_track)]

# plot stress strain for local point
point_df.plot.scatter(x = 'Exx', y = 'stress_mpa', s = 8, c = '#4E598D')
plt.show()

agg_frame_df = all_frames_df.groupby('frame').mean()
# plot average stress-strain response
agg_frame_df.plot.scatter(x = 'Exx', y = 'stress_mpa', s = 8, c = '#4E598D')
plt.show()

#%% Exploratory data analysis
n_bins = 20
frame_range = 36
num_img_x = 6
num_img_y = int(round((frame_range-1)/num_img_x,0))

# ----- plot histogram of strain for each frame -----
plt.style.use('Z:/Python/mpl_styles/stg_plot_style_1.mplstyle')
fig, axs = plt.subplots(num_img_y, num_img_x, sharey=True, tight_layout=True)
    
row, col = 0, 0 # current row and column number for plotting
for i in range(1,frame_range):
    row = int(i/(num_img_x+0.1))
    col = i - row*(num_img_x) - 1
    #print(str(row)+', '+str(col)) # check index logic
    
    # create dataframe for current frame
    single_frame_df = all_frames_df[all_frames_df['frame'] == i]
    
    # calculate measurement area
    # first get average width
    avg_width = single_frame_df.groupby('x_pix').first().mean()['width_mm']
    # get number of pixels in x-direction with valid data
    N = single_frame_df.groupby('x_pix').first().shape[0]
    # calculate area
    area = avg_width*img_scale*N
    
    # We can set the number of bins with the `bins` kwarg
    axs[row,col].hist(single_frame_df['Exx'], edgecolor='#262C47', color = '#4E598D', bins = n_bins, linewidth = 0.5)
    axs[row,col].set_title('Frame: '+ str(i), fontsize = 5)
    axs[row,col].set_xlabel('Exx', fontsize = 5)
    axs[row,col].grid(True, alpha = 0.5)
    axs[row,col].set_xlim([0,1.05*single_frame_df['Exx'].max()])
    axs[row,col].tick_params(labelsize = 5)
    
    # add line showning mean of field at each frame
    axs[row,col].axvline(agg_frame_df.loc[i,'Exx'], color='k', linestyle='dashed', linewidth=0.4, marker = '')

    _, max_ylim = plt.ylim()
    axs[row,col].text(agg_frame_df.loc[i,'Exx']*1.1, max_ylim*0.8, 'Mean: {:.2f}'.format(agg_frame_df.loc[i,'Exx']), fontsize = 4)

plt.show()

# generate boxplot for each frame (Exx, Eyy, Exy)
frame_labels = all_frames_df['frame'].unique().astype(int).astype(str)
mpl.rcParams['lines.marker']=''


data_exx = []
data_eyy = []
data_exy = []
for i in range(0,frame_range):
    data_exx.append(np.array(all_frames_df[all_frames_df['frame'] == i]['Exx']))
    data_eyy.append(np.array(all_frames_df[all_frames_df['frame'] == i]['Eyy']))
    data_exy.append(np.array(all_frames_df[all_frames_df['frame'] == i]['Exy']))

#%%
# mpl.rcParams['font.sans-serif'] = "Avenir LT Std"
# # Then, "ALWAYS use sans-serif fonts"
# mpl.rcParams['font.family'] = "sans-serif"

boxprops = dict(linestyle='-', linewidth=0.5, color = '#4E598D')
flierprops = dict(marker='o', markerfacecolor='#8D4E55', markersize=1,
                  linestyle='-', linewidth = 0.1, markeredgecolor = '#4C161C',
                  markeredgewidth = 0.0, alpha = 0.3)
meanprops = dict(linestyle='-', linewidth=0.5, color = '#8D4E55')
medianprops = dict(linestyle='-', linewidth=0.5, color = '#8D4E55')

def plot_boxplot_vs_frame(data, ylabel):
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

# plot strain box plots for each frame
plot_boxplot_vs_frame(data_exx, ylabel = 'Exx')
plot_boxplot_vs_frame(data_eyy, ylabel = 'Eyy')
plot_boxplot_vs_frame(data_exy, ylabel = 'Exy')

#%%

# load in colormap
cm_data = np.loadtxt('Z:/Python/mpl_styles/'+cmap_name+'.txt')
custom_map = LinearSegmentedColormap.from_list('custom', cm_data)

# ----- find strains in each 'mode' and plot stress-strain response -----
# create dataframe for frame where bins clearly visible
single_frame_df = all_frames_df[all_frames_df['frame'] == 15]

high_strain_df = single_frame_df[(single_frame_df['Exx'] > 1.6) &
                                 (single_frame_df['Exx'] < 1.8)]

low_strain_df = single_frame_df[(single_frame_df['Exx'] > 0.7) &
                                 (single_frame_df['Exx'] < 0.8)]

# plot high (mode 2) strain points
p1 = high_strain_df.plot.scatter(x = 'x_pix', y = 'y_pix', s = 2, c = 'Exx')
plt.xlim([0, 1200])
plt.ylim([800,1150])
plt.tight_layout()
plt.show()

# plot low (mode 1) strain points
low_strain_df.plot.scatter(x = 'x_pix', y = 'y_pix', s = 2, c = 'Exx')
plt.xlim([0, 1200])
plt.ylim([800,1150])
plt.tight_layout()
plt.show()

# ----- plot stress-strain for random samples -----
# select random sample of 15 points from high and low strain regions
high_strain_sample = high_strain_df.sample(n = 15, random_state = 1)
low_strain_sample = low_strain_df.sample(n = 15, random_state = 1)

# extract all points matching coordinates from both samples
agg_high_strain_sample = all_frames_df[
    (all_frames_df['x_pix'].isin(high_strain_sample['x_pix'])) & 
    (all_frames_df['y_pix'].isin(high_strain_sample['y_pix']))]

agg_low_strain_sample = all_frames_df[
    (all_frames_df['x_pix'].isin(low_strain_sample['x_pix'])) & 
    (all_frames_df['y_pix'].isin(low_strain_sample['y_pix']))]

# generate scatter plots
f = plt.figure(figsize = (3,3))
ax = f.add_subplot(1,1,1)
ax.scatter(agg_high_strain_sample['frame'], agg_high_strain_sample['Exx'],
           s = 3, c = '#4E598D', edgecolors = '#262C47', 
           linewidths = 0.5, label = '1.6 < Exx < 1.8')
ax.scatter(agg_low_strain_sample['frame'], agg_low_strain_sample['Exx'],
           s = 3, c = '#8D4E55', edgecolors = '#4C161C',
           linewidths = 0.5, label = '0.7 < Exx < 0.85')
ax.grid(True, alpha = 0.5)
ax.set_xlim([0, len(files_gom)+2])
ax.set_xlabel('Frame', fontsize = 6)
ax.set_ylabel('Exx', fontsize = 6)
plt.legend(loc='upper left', fontsize = 6)
plt.tight_layout()
plt.show()
'''
agg_high_strain_sample.plot.scatter(x = 'Exx', y = 'stress_mpa', s = 8, c = '#4E598D')
plt.grid(True, alpha = 0.5)
plt.show()

agg_low_strain_sample.plot.scatter(x = 'Exx', y = 'stress_mpa', s = 8, c = '#8D4E55')
plt.grid(True, alpha = 0.5)
plt.show()
'''


