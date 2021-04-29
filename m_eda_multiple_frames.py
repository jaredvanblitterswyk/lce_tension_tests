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

#%% Functions
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

#%% ----- MAIN SCRIPT -----
# ----- configure directories -----
# root directory
dir_root = 'Z:/Experiments/lce_tension'
# extensions to access sub-directories
batch_ext = 'lcei_001'
mts_ext = 'mts_data'
sample_ext = '002_t02_r00'
gom_ext = 'gom_results'
cmap_name = 'lajolla' # custom colormap stored in mpl_styles

# load single frames flag
load_multiple_frames = True

# load in colormap
cm_data = np.loadtxt('Z:/Python/mpl_styles/'+cmap_name+'.txt')
custom_map = LinearSegmentedColormap.from_list('custom', cm_data)

# define full paths to mts and gom data
dir_xsection = os.path.join(dir_root,batch_ext,sample_ext)
dir_mts = os.path.join(dir_root,batch_ext,mts_ext,batch_ext+'_'+sample_ext)
dir_gom_results = os.path.join(dir_root,batch_ext,sample_ext,gom_ext)

img_scale = 0.0106 # image scale (mm/pix)
if load_multiple_frames:
    for i in range(1,35):
        print('Adding frame: '+str(i))
        save_filename = 'results_df_frame_' + '{:02d}'.format(i) + '.pkl'
        frame_df = pd.read_pickle(os.path.join(dir_gom_results,save_filename))
        
        # add time stamp to frame to allow for sorting later
        frame_df['frame'] = i*np.ones((frame_df.shape[0],))
        
        if i == 1:
            # create empty data frame to store all values from each frame
            all_frames_df = pd.DataFrame(columns = frame_df.columns)
        
        all_frames_df = pd.concat(
            [all_frames_df, frame_df], 
            axis = 0, join = 'outer'
            )
        
    # filter data to track a specific point
    x_track, y_track = 320, 875 # pixel coordinates of point to track
    
    point_df = all_frames_df[
        (all_frames_df['x_pix'] == x_track) 
        & (all_frames_df['y_pix'] == y_track)
        ]

'''
# plot stress strain for local point
point_df.plot.scatter(x = 'Exx', y = 'stress_mpa', s = 8, c = '#4E598D')
plt.show()

agg_frame_df = all_frames_df.groupby('frame').mean()
# plot average stress-strain response
agg_frame_df.plot.scatter(x = 'Exx', y = 'stress_mpa', s = 4, c = '#4E598D')
plt.show()
'''

all_frames_df = all_frames_df.dropna(axis = 0)

#%% ----- EXPLORATORY DATA ANALYSIS -----
# ----------------------------------------------------------------------------
# ----- plot histogram and box plot of strain for each frame -----
# ----------------------------------------------------------------------------
n_bins = 20
frame_range = len(
    [f for f in os.listdir(dir_gom_results) if f.endswith('.pkl')]
    )

# set frame range manually
frame_range = 39

num_img_x = 6
num_img_y = int(round((frame_range)/num_img_x,0))

# define custom plot style
plt.style.use('Z:/Python/mpl_styles/stg_plot_style_1.mplstyle')
fig, axs = plt.subplots(num_img_y, num_img_x, sharey=True, tight_layout=True)
    
row, col = 0, 0 # current row and column number for plotting

# create list of arrays with strain values at each frame for boxplots
data_exx = []
data_eyy = []
data_exy = []

var_to_plot = 'Eyy'

# loop through range of frames and generate histogram and box plots
plot_num = 0

for i in range(3,frame_range):
    row = int(plot_num/(num_img_x))
    col = plot_num - row*(num_img_x)
    #print(str(row)+', '+str(col)) # check index logic
    
    # create dataframe for current frame
    if load_multiple_frames: 
        single_frame_df = all_frames_df[all_frames_df['frame'] == i]
    else:
        print('Processing frame: '+str(i))
        save_filename = 'results_df_frame_'+str(i)+'.pkl'
        single_frame_df = pd.read_pickle(
            os.path.join(dir_gom_results,save_filename)
            )
    
    # calculate measurement area
    avg_width = single_frame_df.groupby('x_pix').first().mean()['width_mm']
    N = single_frame_df.groupby('x_pix').first().shape[0]
    area = avg_width*img_scale*N
    
    # We can set the number of bins with the `bins` kwarg
    axs[row,col].hist(
        single_frame_df[var_to_plot], 
        edgecolor='#262C47', color = '#4E598D', 
        bins = n_bins, linewidth = 0.5
        )
    axs[row,col].set_title('Frame: '+ str(i), fontsize = 5)
    axs[row,col].set_xlabel(var_to_plot, fontsize = 5)
    axs[row,col].grid(True, alpha = 0.5)
    axs[row,col].set_xlim(
        [1.05*single_frame_df[var_to_plot].min(),
         1.05*single_frame_df[var_to_plot].max()
         ]
        )
    #axs[row,col].set_xlim([0,1.05*single_frame_df['Eyy'].max()])
    axs[row,col].tick_params(labelsize = 5)
    
    # extract ylims of plot for annotations
    _, max_ylim = plt.ylim()
    
    # add line showning mean of field at each frame
    if load_multiple_frames:
        avg_strain = all_frames_df[
            all_frames_df['frame'] == i
            ][var_to_plot].mean()
        axs[row,col].axvline(
            avg_strain,
            color='k', linestyle='dashed', linewidth=0.4, marker = ''
            )
        axs[row,col].text(
            avg_strain*1.1, max_ylim*0.8, 
            'Mean: {:.2f}'.format(avg_strain), fontsize = 4
            ) 
    else:
        #calculate average strain
        avg_strain = single_frame_df[var_to_plot].mean()
        axs[row,col].axvline(
            avg_strain, 
            color='k', linestyle='dashed', linewidth=0.4, marker = ''
            )
        axs[row,col].text(
            avg_strain*1.1, max_ylim*0.8, 
            'Mean: {:.2f}'.format(avg_strain), fontsize = 4
            )
        
    # before removing frame data from memory, store strains in arrays for box plots
    if not load_multiple_frames:
        data_exx.append(np.array(single_frame_df['Exx']))
        data_eyy.append(np.array(single_frame_df['Eyy']))
        data_exy.append(np.array(single_frame_df['Exy']))    

    # if loading each file individually, delete after use
    del single_frame_df
    plot_num += 1
    
plt.show()

#%% ----- Generate Box Plots -----
# if loading all frames at once, aggregate strain data into list for plotting
if load_multiple_frames: 
    frame_labels = all_frames_df['frame'].unique().astype(int).astype(str)
    
    # append strains to list for boxplots
    for i in range(0,frame_range):
        data_exx.append(
            np.array(
                all_frames_df[all_frames_df['frame'] == i]['Exx']
                )
            )
        data_eyy.append(
            np.array(
                all_frames_df[all_frames_df['frame'] == i]['Eyy']
                )
            )
        data_exy.append(
            np.array(
                all_frames_df[all_frames_df['frame'] == i]['Exy']
                )
            )
else:
    # data aggregation already complete, just calculate frame labels
    frame_labels = np.linspace(1, frame_range, frame_range).astype(int).astype(str)

# ----- generate box plots -----
mpl.rcParams['lines.marker']=''
plot_boxplot_vs_frame(data_exx, ylabel = 'Exx')
plot_boxplot_vs_frame(data_eyy, ylabel = 'Eyy')
plot_boxplot_vs_frame(data_exy, ylabel = 'Exy')

#%%
# ----------------------------------------------------------------------------
# ----- filter data into strain ranges based on selected frame -----
# ----------------------------------------------------------------------------
# create dataframe for frame used to define coordinates in strain bands 
mask_frame = 20
num_category_bands = 6
y_var = 'R'
x_var = 'frame'
cat_var = 'Exx'
num_samples_to_plot = 4000

single_frame_df = all_frames_df[all_frames_df['frame'] == mask_frame]

# calculate strain range bounds
max_category_band = round(single_frame_df[cat_var].quantile(0.98),1)
min_category_band = round(single_frame_df[cat_var].min(),1)

category_ranges = np.linspace(min_category_band, max_category_band, num_category_bands)

# colour and edge colour arrays (hard coded to 10 or less strain bands)
ec = ['#003057', '#00313C', '#9E2A2F', '#623412','#284723',
      '#59315F', '#A45A2A', '#53565A', '#007377', '#453536']
c = ['#BDD6E6', '#8DB9CA', '#F4C3CC', '#D9B48F', '#9ABEAA',
     '#C9B1D0', '#F1C6A7', '#C8C9C7', '#88DBDF', '#C1B2B6']

num_img_x = 3
num_img_y = int((num_category_bands)/num_img_x)

fig, axs = plt.subplots(num_img_y, num_img_x, figsize=(5,5),
                        sharey=True, tight_layout=True)

# generate plot (either load in all at once, or individually - memory issues)
if load_multiple_frames:
    single_frame_df = all_frames_df[all_frames_df['frame'] == mask_frame]
    
    for i in range(1,num_category_bands+1):
        row = int(i/(num_img_x+0.1))
        col = i - row*(num_img_x) - 1
        #print(str(row)+', '+str(col)) # check index logic
    
        if i == num_category_bands:
            category_band_df = single_frame_df[
                single_frame_df[cat_var] >= category_ranges[i-1]]   
        else:
            category_band_df = single_frame_df[(
                single_frame_df[cat_var] >= category_ranges[i-1]
                ) & (
                single_frame_df[cat_var] < category_ranges[i]
                )]
                
        # find all points within that strain range
        agg_strain_band_df = all_frames_df[
        all_frames_df.index.isin(category_band_df.index)]
        
        if agg_strain_band_df.shape[0] > num_samples_to_plot:
            category_band_sample = agg_strain_band_df.sample(
                n = num_samples_to_plot, random_state = 1
                )
        else:
            category_band_sample = agg_strain_band_df.copy()
        
        # add to plot
        axs[row,col].scatter(
            category_band_sample[x_var], 
            category_band_sample[y_var],
            s = 2, c = c[i-1], edgecolors = ec[i-1], 
            alpha = 0.4, linewidths = 0.5
            )
        axs[row,col].set_ylim([0, round(
            all_frames_df[y_var].quantile(0.995),1
            )
            +0.5
            ])
        #axs[row,col].set_xlim([0,len(all_frames_df[dependent_var].unique())+2])
        axs[row,col].set_xlim([0,round(
            all_frames_df[x_var].quantile(0.995),1
            )
            +0.5
            ])
        axs[row,col].set_ylabel(y_var)
        axs[row,col].set_xlabel(x_var)
        axs[row,col].grid(True, alpha = 0.5,zorder = 0)
        if i == num_strain_bands:
            axs[row,col].set_title(
                cat_var+'_band: '+'>'+str(round(category_ranges[i-1],1)),
                fontsize = 5
                )
        else:       
            axs[row,col].set_title(
                cat_var+'_band: '+str(round(category_ranges[i-1],1))+
                ':'+ str(round(category_ranges[i],1)), 
                fontsize = 5
                )
    
        # add line showning mean of field at each frame
        axs[row,col].axvline(mask_frame, linestyle='dashed', 
                             linewidth=0.4, marker = '')
    
        _, max_ylim = plt.ylim()
        axs[row,col].text(mask_frame*1.1, max_ylim*0.8,
                          'Mask frame: {:.0f}'.format(mask_frame),
                          fontsize = 4)
    
else:
    save_filename = 'results_df_frame_'+str(mask_frame)+'.pkl'
    single_frame_df = pd.read_pickle(
        os.path.join(dir_gom_results,save_filename)
        )

    # create a list of indices corresponding to each strain band
    strain_band_indices = []
    for i in range(1,num_strain_bands+1):
        if i == num_strain_bands:
            category_band_df = single_frame_df[
                single_frame_df[y_var] >= category_ranges[i-1]]   
        else:
            category_band_df = single_frame_df[(
                single_frame_df[y_var] >= category_ranges[i-1]
                ) & (
                single_frame_df[y_var] < strain_ranges[i]
                )]
                    
        category_band_indices.append(category_band_df.index)

    # ----- plot strain bands vs time -----
    for j in range(0,frame_range):
        print('Plotting frame: '+str(j))
        save_filename = 'results_df_frame_'+str(j)+'.pkl'
        curr_frame_df = pd.read_pickle(
            os.path.join(dir_gom_results,save_filename)
            )
        
        # add time stamp to frame to allow for sorting later
        curr_frame_df['frame'] = j*np.ones((curr_frame_df.shape[0],))      
        row, col = 0, 0 # current row and column number for plotting
        for i in range(1,num_category_bands+1):
            row = int(i/(num_img_x+0.1))
            col = i - row*(num_img_x) - 1
            #print(str(row)+', '+str(col)) # check index logic
                    
            # find all points within that strain range
            category_band = curr_frame_df[
                curr_frame_df.index.isin(category_band_indices[i-1])
                ]
                                      
            
            if category_band.shape[0] > num_samples_to_plot:
                category_band_sample = category_band.sample(
                    n = num_samples_to_plot, random_state = 1
                    )
            else:
                category_band_sample = category_band.copy()
            
            # add to plot
            axs[row,col].scatter(
                category_band_sample[x_var], 
                category_band_sample[y_var],
                s = 2, c = c[i-1], edgecolors = ec[i-1], 
                alpha = 0.4, linewidths = 0.5
                )
            
            axs[row,col].set_xlim([0,frame_range+2])
            axs[row,col].set_ylim([min_category_band-0.1,max_category_band+0.1])
            axs[row,col].set_xlabel(x_var)
            axs[row,col].set_ylabel(y_var)
            axs[row,col].grid(True, alpha = 0.5,zorder = 0)
            if i == num_category_bands:
                axs[row,col].set_title(
                    cat_var+'_band: '+'>'+str(round(category_ranges[i-1],1)),
                    fontsize = 5
                    )
            else:       
                axs[row,col].set_title(
                    cat_var+'_band: '+str(round(category_ranges[i-1],1))+
                    ':'+ str(round(category_ranges[i],1)), 
                    fontsize = 5
                    )
            
            # add line showning mean of field at each frame
            axs[row,col].axvline(mask_frame, linestyle='dashed', 
                                 linewidth=0.4, marker = '')
        
            _, max_ylim = plt.ylim()
            axs[row,col].text(mask_frame*1.1, max_ylim*0.8,
                              'Mask frame: {:.0f}'.format(mask_frame),
                              fontsize = 4)    

#%% ----- STRESS VS. EXX (STRAIN BANDS) -----
f = plt.figure(figsize = (3,3))
ax = f.add_subplot(1,1,1)
for i in range(4,num_strain_bands+1):
    if i == num_strain_bands:
        strain_band_df = single_frame_df[
            single_frame_df['Exx'] >= strain_ranges[i-1]]   
    else:
        strain_band_df = single_frame_df[(
            single_frame_df['Exx'] >= strain_ranges[i-1]
            ) & (
            single_frame_df['Exx'] < strain_ranges[i]
            )]
            
    # find all points within that strain range
    agg_strain_band_df = all_frames_df[
    all_frames_df.index.isin(strain_band_df.index)]
    
    if agg_strain_band_df.shape[0] > 2000:
        strain_band_sample = agg_strain_band_df.sample(n = 2000, 
                                                       random_state = 1)
    else:
        strain_band_sample = agg_strain_band_df.copy()
    
    # add to plot
    ax.scatter(
        strain_band_sample['Exx'], 
        strain_band_sample['stress_mpa'],
        s = 2, 
        c = c[i-1],
        edgecolors = ec[i-1], 
        alpha = 0.4,
        linewidths = 0.5, 
        label = 'strain_band: '+str(i)
        )
    
    ax.set_xlim([0,round(all_frames_df['Exx'].max(),1)+0.5])
    ax.set_ylim([0,round(all_frames_df['stress_mpa'].max(),1)+0.05])
    ax.grid(True, alpha = 0.5,zorder = 0)
    ax.set_xlabel('Exx')
    ax.set_ylabel('Stress (Mpa)')
plt.legend(loc='upper left', fontsize = 4)

legend = ax.legend(fontsize = 4)
legend.get_frame().set_linewidth(0.5)

#%% ----- OVERLAY STRAIN BAND LOCATIONS ON UNDEFORMED SAMPLE -----
single_frame_df = all_frames_df[all_frames_df['frame'] == mask_frame]

f = plt.figure(figsize = (5,1.3))
ax = f.add_subplot(1,1,1)
for i in range(1,num_strain_bands+1):
    if i == num_strain_bands:
            strain_band_df = single_frame_df[
                single_frame_df['Exx'] >= strain_ranges[i-1]]   
    else:
        strain_band_df = single_frame_df[(
            single_frame_df['Exx'] >= strain_ranges[i-1]
            ) & (
            single_frame_df['Exx'] < strain_ranges[i]
            )]
    
    incl_strain_band_df = all_frames_df[
        all_frames_df.index.isin(strain_band_df.index)]
    
    # plot on one figure
    if i == 1:
        ax.scatter(
            all_frames_df[all_frames_df['frame'] == 1][['x_pix']]*img_scale, 
            all_frames_df[all_frames_df['frame'] == 1][['y_pix']]*img_scale, 
            s = 1, c = '#D0D3D4', edgecolors = '#D0D3D4', 
            alpha = 0.3, linewidths = 0, zorder = 0, label = 'all_other_points'
            )
    else:
        ax.scatter(
            incl_strain_band_df[incl_strain_band_df['frame'] == 1][['x_pix']]*img_scale, 
            incl_strain_band_df[incl_strain_band_df['frame'] == 1][['y_pix']]*img_scale, 
            s = 1, c = c[i-1], edgecolors = ec[i-1], alpha = 1, 
            linewidths = 0, zorder = i, label = 'Range: '+str(i)
            )

# ----- customize plot - add grid, labels, and legend -----
ax.grid(True, alpha = 0.5)
ax.set_xlabel('x (mm)')
ax.set_ylabel('y (mm)')
ax.set_xlim([0,15])

# shrink current axes box to place legend overhead with axes labels
# Shrink current axis's height by 10% on the bottom
box = ax.get_position()
ax.set_position(
    [box.x0, box.y0 + box.height * 0.1,
     box.width, box.height * 0.7]
    )

legend = ax.legend(
    fontsize = 3, loc='upper center', ncol=num_strain_bands+1, 
    bbox_to_anchor=(0.5, 1.12)
    )
legend.get_frame().set_linewidth(0.5)
plt.tight_layout()

#%% ----- PLOT WIDTH-AVERAGED STRESS VS STRAIN -----
f = plt.figure(figsize = (3,3))
ax = f.add_subplot(1,1,1)
for i in range(1,frame_range):
    df = all_frames_df[all_frames_df['frame'] == i]

    # group by x pix and take average of stress and strain
    df_xs_group = df.groupby('x_pix').mean()

    ax.scatter(df_xs_group[['Exx']], 
               df_xs_group[['stress_mpa']], 
               s = 1, 
               c = '#BDD6E6', 
               edgecolors = '#003057', 
               alpha = 0.4,
               linewidths = 0.5)

ax.set_xlim([0,round(all_frames_df['Exx'].max(),1)-1])
ax.set_ylim([0,round(all_frames_df['stress_mpa'].max(),1)-0.05])
ax.grid(True, alpha = 0.5,zorder = 0)
ax.set_xlabel('Exx')
ax.set_ylabel('Stress (Mpa)')

#%% ----- Plot 3D SCATTER (STRAIN COMPONENTS VS ROTATION) ---
all_frames_df['x_mm'] = all_frames_df['x_pix']*img_scale + all_frames_df['ux']
all_frames_df['y_mm'] = all_frames_df['y_pix']*img_scale + all_frames_df['uy']

# filter based on frame and location
xx = np.array(all_frames_df[(all_frames_df['frame'] == mask_frame) & 
                            (all_frames_df['x_mm'] > 15) &
                            (all_frames_df['x_mm'] < 40) &
                            (all_frames_df['R'] > 1)][['Exx']])

yy = np.array(all_frames_df[(all_frames_df['frame'] == mask_frame) & 
                            (all_frames_df['x_mm'] > 15) &
                            (all_frames_df['x_mm'] < 40) &
                            (all_frames_df['R'] > 1)][['Eyy']])

zz = np.array(all_frames_df[(all_frames_df['frame'] == mask_frame) & 
                            (all_frames_df['x_mm'] > 15) &
                            (all_frames_df['x_mm'] < 40) & 
                            (all_frames_df['R'] > 1)][['Exy']])

cc = np.array(all_frames_df[(all_frames_df['frame'] == mask_frame) & 
                            (all_frames_df['x_mm'] > 15) &
                            (all_frames_df['x_mm'] < 40)&
                            (all_frames_df['R'] > 1)][['R']])

f = plt.figure(figsize = (4,3))
ax = plt.axes(projection ="3d")
 
# Creating plot
scttr = ax.scatter3D(xx, yy, zz, c = cc, cmap = custom_map, s = 0.1, vmin = -1, vmax = 4)
ax.set_xlabel('Exx')
ax.set_ylabel('Eyy')
ax.set_zlabel('Exy')
ax.grid(True, alpha = 0.5)
ax.set_title('Frame 20')
f.colorbar(scttr, ax = ax, shrink = 0.5, aspect = 10)
plt.tight_layout()
plt.show()

#%% ----- EXX VS EYY (CHECK COMPRESSIBILITY) -----

# ----------------------------------------------------------------------------
# ----- filter data into strain ranges based on selected frame -----
# ----------------------------------------------------------------------------
# --- plot setup ---
mask_frame = 20
num_category_bands = 6
y_var = 'Eyy'
x_var = 'Exx'
cat_var = 'Exx'
num_samples_to_plot = 4000

# colour and edge colour arrays (hard coded to 10 or less strain bands)
ec = ['#003057', '#00313C', '#9E2A2F', '#623412','#284723',
      '#59315F', '#A45A2A', '#53565A', '#007377', '#453536']
c = ['#BDD6E6', '#8DB9CA', '#F4C3CC', '#D9B48F', '#9ABEAA',
     '#C9B1D0', '#F1C6A7', '#C8C9C7', '#88DBDF', '#C1B2B6']

x_fit = np.linspace(0,2.8,500)
y_fit = 0.5*(1/np.sqrt(1+2*x_fit) - 1)

num_img_x = 1
num_img_y = 1

# ----- processing/plotting -----
single_frame_df = all_frames_df[all_frames_df['frame'] == mask_frame]

# calculate strain range bounds
max_category_band = round(single_frame_df[cat_var].quantile(0.98),1)
min_category_band = round(single_frame_df[cat_var].min(),1)

category_ranges = np.linspace(
    min_category_band, max_category_band, num_category_bands
    )

fig, axs = plt.subplots(num_img_x, num_img_y, figsize=(3,3),
                        sharey=True, tight_layout=True)

for i in range(4,num_category_bands+1):

    if i == num_category_bands:
        category_band_df = single_frame_df[
            single_frame_df[cat_var] >= category_ranges[i-1]]   
    else:
        category_band_df = single_frame_df[(
            single_frame_df[cat_var] >= category_ranges[i-1]
            ) & (
            single_frame_df[cat_var] < category_ranges[i]
            )]
            
    # find all points within that strain range
    agg_strain_band_df = all_frames_df[
    all_frames_df.index.isin(category_band_df.index)]
    
    if agg_strain_band_df.shape[0] > num_samples_to_plot:
        category_band_sample = agg_strain_band_df.sample(
            n = num_samples_to_plot, random_state = 1
            )
    else:
        category_band_sample = agg_strain_band_df.copy()
    
    # add to plot
    axs.scatter(
        category_band_sample[x_var], 
        category_band_sample[y_var],
        s = 2, c = c[i-1], edgecolors = ec[i-1], 
        alpha = 0.3, linewidths = 0.5, label = 'Band: ' + str(i)
        )
    '''
    axs.set_ylim([0, round(
        all_frames_df[y_var].quantile(0.995),1
        )
        +0.5
        ])
    #axs[row,col].set_xlim([0,len(all_frames_df[dependent_var].unique())+2])
    axs.set_xlim([0,round(
        all_frames_df[x_var].quantile(0.995),1
        )
        +0.5
        ])
    '''
    axs.set_ylabel(y_var)
    axs.set_xlabel(x_var)
    axs.grid(True, alpha = 0.5,zorder = 0)  

# compare against incompressible relationship    
axs.scatter(
    x_fit, y_fit, s = 0.5, c = 'k', edgecolors = 'k', 
    alpha = 1.0, label = 'Incompressible')

# add legend
plt.legend(loc='upper right', fontsize = 4)
legend = axs.legend(fontsize = 4)
legend.get_frame().set_linewidth(0.5)

#%% ----- EXTRA PLOT CAPABILITIES -----
# ----------------------------------------------------------------------------
# Take last group and overlay spatial mask
'''
strain_band = 8

if strain_band == num_strain_bands:
        strain_band_df = single_frame_df[
            single_frame_df['Exx'] >= strain_ranges[strain_band-1]]   
else:
    strain_band_df = single_frame_df[(
        single_frame_df['Exx'] >= strain_ranges[strain_band-1]
        ) & (
        single_frame_df['Exx'] < strain_ranges[strain_band]
        )]

incl_strain_band_df = all_frames_df[
    all_frames_df.index.isin(strain_band_df.index)]
        
excl_strain_band_df = all_frames_df[
    all_frames_df.index.isin(strain_band_df.index) == False]

# plot on one figure
f = plt.figure(figsize = (3,1))
ax = f.add_subplot(1,1,1)
ax.scatter(incl_strain_band_df[incl_strain_band_df['frame'] == 0][['x_pix']], 
               incl_strain_band_df[incl_strain_band_df['frame'] == 0][['y_pix']], 
               s = 1, 
               c = '#584446', 
               edgecolors = '#584446', 
               alpha = 0.4,
               linewidths = 0, zorder = 0)

ax.scatter(excl_strain_band_df[excl_strain_band_df['frame'] == 0][['x_pix']], 
               excl_strain_band_df[excl_strain_band_df['frame'] == 0][['y_pix']], 
               s = 1, 
               c = '#D0D3D4', 
               edgecolors = '#D0D3D4', 
               alpha = 0.4,
               linewidths = 0, zorder = 1)
ax.grid(True, alpha = 0.5)
ax.set_xlabel('x (pix)')
ax.set_ylabel('y (pix)')
ax.set_xlim([0,1500])
if strain_band == num_strain_bands:
    ax.set_title('strain_band: '+'>'+str(round(strain_ranges[strain_band-1],1)),
                          fontsize = 5)
else:       
    ax.set_title('strain_band: '+str(round(strain_ranges[strain_band-1],1))+
                       ':'+ str(round(strain_ranges[strain_band],1)), fontsize = 5)
plt.tight_layout()

# ----- MASk MEASUREMENT POINTS BY STRAIN RANGE (ON SPECIMEN) -----
# hard code strain rnages based on results from plot of strain band vs. frame

exx_max = 0.7
exy_min = -0.15
exy_max = 0.15

if ~load_multiple_frames:
    save_filename = 'results_df_frame_'+str(mask_frame)+'.pkl'
    single_frame_df = pd.read_pickle(
        os.path.join(dir_gom_results,save_filename)
        )

    # filter based on strain values 
    strain_band_exy_df = single_frame_df[(
        single_frame_df['Exy'] >= exy_max
        ) | (
        single_frame_df['Exy'] < exy_min
        )
    ]   

    strain_band_exx_df = single_frame_df[
        single_frame_df['Exx'] >= exx_max
        ]
    
    excl_strain_band_df = single_frame_df[(
        single_frame_df.index.isin(strain_band_exy_df.index) == False)
        & (
        single_frame_df.index.isin(strain_band_exx_df.index) == False)
        ]
    
    # plot on one figure
    f = plt.figure(figsize = (2,3))
    ax = f.add_subplot(1,1,1)
    ax.scatter(
        strain_band_exx_df[['x_pix']], 
        strain_band_exx_df[['y_pix']], 
        s = 1, 
        c = '#003057', 
        edgecolors = '#003057', 
        alpha = 0.4,
        linewidths = 0, zorder = 0
        )
    ax.scatter(
        strain_band_exy_df[['x_pix']], 
        strain_band_exy_df[['y_pix']], 
        s = 1, 
        c = '#9E2A2F', 
        edgecolors = '#9E2A2F', 
        alpha = 0.4,
        linewidths = 0, zorder = 1
        )
    ax.scatter(
        excl_strain_band_df[['x_pix']], 
        excl_strain_band_df[['y_pix']], 
        s = 1, 
        c = '#D0D3D4', 
        edgecolors = '#D0D3D4', 
        alpha = 0.4,
        linewidths = 0, zorder = -1
        )
    ax.grid(True, alpha = 0.5)
    ax.set_xlabel('x (pix)')
    ax.set_ylabel('y (pix)')
    ax.set_title('Exx > '+str(exx_max)+', '+str(exy_min)+'< Exy <= '+str(exy_max))
    
    plt.tight_layout()
'''
