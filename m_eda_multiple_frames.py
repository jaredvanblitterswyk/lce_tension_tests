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
sample_ext = '006_t02_r00'
gom_ext = 'gom_results'
cmap_name = 'lajolla' # custom colormap stored in mpl_styles

# load single frames flag
load_multiple_frames = False

# load in colormap
cm_data = np.loadtxt('Z:/Python/mpl_styles/'+cmap_name+'.txt')
custom_map = LinearSegmentedColormap.from_list('custom', cm_data)

# define full paths to mts and gom data
dir_xsection = os.path.join(dir_root,batch_ext,sample_ext)
dir_mts = os.path.join(dir_root,batch_ext,mts_ext,batch_ext+'_'+sample_ext)
dir_gom_results = os.path.join(dir_root,batch_ext,sample_ext,gom_ext)

img_scale = 0.0187 # image scale (mm/pix)
if load_multiple_frames:
    for i in range(0,46):
        print('Adding frame: '+str(i))
        save_filename = 'results_df_frame_'+str(i)+'.pkl'
        frame_df = pd.read_pickle(os.path.join(dir_gom_results,save_filename))
        
        # add time stamp to frame to allow for sorting later
        frame_df['frame'] = i*np.ones((frame_df.shape[0],))
        
        if i == 0:
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
agg_frame_df.plot.scatter(x = 'Exx', y = 'stress_mpa', s = 8, c = '#4E598D')
plt.show()
'''

#%% ----- EXPLORATORY DATA ANALYSIS -----
# ----------------------------------------------------------------------------
# ----- plot histogram and box plot of strain for each frame -----
# ----------------------------------------------------------------------------
n_bins = 20
frame_range = len(
    [f for f in os.listdir(dir_gom_results) if f.endswith('.pkl')]
    )
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

# loop through range of frames and generate histogram and box plots
for i in range(1,frame_range):
    row = int(i/(num_img_x+0.1))
    col = i - row*(num_img_x) - 1
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
        single_frame_df['Exx'], 
        edgecolor='#262C47', color = '#4E598D', 
        bins = n_bins, linewidth = 0.5
        )
    axs[row,col].set_title('Frame: '+ str(i), fontsize = 5)
    axs[row,col].set_xlabel('Exx', fontsize = 5)
    axs[row,col].grid(True, alpha = 0.5)
    axs[row,col].set_xlim([0,1.05*single_frame_df['Exx'].max()])
    axs[row,col].tick_params(labelsize = 5)
    
    # extract ylims of plot for annotations
    _, max_ylim = plt.ylim()
    
    # add line showning mean of field at each frame
    if load_multiple_frames:
        axs[row,col].axvline(
            agg_frame_df.loc[i,'Exx'],
            color='k', linestyle='dashed', linewidth=0.4, marker = ''
            )
        axs[row,col].text(
            agg_frame_df.loc[i,'Exx']*1.1, max_ylim*0.8, 
            'Mean: {:.2f}'.format(agg_frame_df.loc[i,'Exx']), fontsize = 4
            ) 
    else:
        #calculate average strain
        avg_strain = single_frame_df['Exx'].mean()
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
plt.show()

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
# create dataframe for frame where bins clearly visible
mask_frame = 15
single_frame_df = all_frames_df[all_frames_df['frame'] == mask_frame]

max_strain_band= round(single_frame_df['Exx'].max(),1)
num_strain_bands = 9

strain_ranges = np.linspace(0.2, max_strain_band, num_strain_bands)

# colour and edge colour arrays (hard coded to 10 or less strain bands)
ec = ['#003057', '#00313C', '#9E2A2F', '#623412','#284723',
      '#59315F', '#A45A2A', '#53565A', '#007377', '#453536']
c = ['#BDD6E6', '#8DB9CA', '#F4C3CC', '#D9B48F', '#9ABEAA',
     '#C9B1D0', '#F1C6A7', '#C8C9C7', '#88DBDF', '#C1B2B6']

# plot on one figure
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
    ax.scatter(strain_band_sample['Exx'], 
               strain_band_sample['stress_mpa'],
               s = 2, 
               c = c[i-1],
               edgecolors = ec[i-1], 
               alpha = 0.4,
               linewidths = 0.5, 
               label = 'strain_band: '+str(i))
    ax.set_xlim([0,round(all_frames_df['Exx'].max(),1)+0.5])
    ax.set_ylim([0,round(all_frames_df['stress_mpa'].max(),1)+0.05])
    ax.grid(True, alpha = 0.5,zorder = 0)
    ax.set_xlabel('Exx')
    ax.set_ylabel('Stress (Mpa)')
plt.legend(loc='upper left', fontsize = 4)
legend = ax.legend(fontsize = 4)
legend.get_frame().set_linewidth(0.5)
'''
# add line showning mean of field at each frame
ax.axvline(mask_frame, linestyle='dashed', linewidth=0.4, marker = '')
_, max_ylim = plt.ylim()
ax.text(mask_frame*1.1, max_ylim*0.8, 
        'Mask: {:.0f}'.format(mask_frame), 
        fontsize = 4)
'''

#%%    
# generate sub plots for each strain band
num_img_x = 3
num_img_y = int((num_strain_bands)/num_img_x)

fig, axs = plt.subplots(num_img_y, num_img_x, figsize=(5,5),
                        sharey=True, tight_layout=True)
    
row, col = 0, 0 # current row and column number for plotting
for i in range(1,num_strain_bands+1):
    row = int(i/(num_img_x+0.1))
    col = i - row*(num_img_x) - 1
    #print(str(row)+', '+str(col)) # check index logic

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
    axs[row,col].scatter(strain_band_sample['frame'], 
                         strain_band_sample['Exx'],
                         s = 2, 
                         c = c[i-1], 
                         edgecolors = ec[i-1], 
                         alpha = 0.4,
                         linewidths = 0.5)
    axs[row,col].set_xlim([0,len(all_frames_df['frame'].unique())+2])
    axs[row,col].set_ylim([0,round(all_frames_df['Exx'].max(),1)+0.5])
    axs[row,col].set_xlabel('Frame')
    axs[row,col].set_ylabel('Exx')
    axs[row,col].grid(True, alpha = 0.5,zorder = 0)
    if i == num_strain_bands:
        axs[row,col].set_title('strain_band: '+'>'+str(round(strain_ranges[i-1],1)),
                              fontsize = 5)
    else:       
        axs[row,col].set_title('strain_band: '+str(round(strain_ranges[i-1],1))+
                           ':'+ str(round(strain_ranges[i],1)), fontsize = 5)
    
    # add line showning mean of field at each frame
    axs[row,col].axvline(mask_frame, linestyle='dashed', 
                         linewidth=0.4, marker = '')

    _, max_ylim = plt.ylim()
    axs[row,col].text(mask_frame*1.1, max_ylim*0.8,
                      'Mask frame: {:.0f}'.format(mask_frame),
                      fontsize = 4)
    
#%% Take last group and overlay spatial mask - run after cell above
strain_band = 9

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
#%%
f = plt.figure(figsize = (4,5))
ax = f.add_subplot(1,1,1)
sc = ax.scatter(
    single_frame_df['x_pix']*img_scale+single_frame_df['ux'], 
    single_frame_df['y_pix']*img_scale+single_frame_df['uy'], 
    s = 1, 
    c = single_frame_df['Exx'],  
    alpha = 1,
    linewidths = 0)
ax.grid(True, alpha = 0.5)
ax.set_xlabel('x (mm)')
ax.set_ylabel('y (mm)')
#ax.set_xlim([0,1500])
f.colorbar(sc, ax=ax)
plt.tight_layout()

#%% Overlay locations of strain bands on undeformed image

f = plt.figure(figsize = (3,1))
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
    
    '''        
    excl_strain_band_df = all_frames_df[
        all_frames_df.index.isin(strain_band_df.index) == False]
    '''
    
    # plot on one figure
    if i == 1:
        ax.scatter(all_frames_df[all_frames_df['frame'] == 0][['x_pix']], 
                       all_frames_df[all_frames_df['frame'] == 0][['y_pix']], 
                       s = 1, 
                       c = '#D0D3D4', 
                       edgecolors = '#D0D3D4', 
                       alpha = 0.3,
                       linewidths = 0, zorder = 0, label = 'all_points')
    
    ax.scatter(incl_strain_band_df[incl_strain_band_df['frame'] == 0][['x_pix']], 
                   incl_strain_band_df[incl_strain_band_df['frame'] == 0][['y_pix']], 
                   s = 1, 
                   c = c[i-1], 
                   edgecolors = ec[i-1], 
                   alpha = 0.8,
                   linewidths = 0, zorder = i, label = 'Range: '+str(i))
ax.grid(True, alpha = 0.5)
ax.set_xlabel('x (pix)')
ax.set_ylabel('y (pix)')
ax.set_xlim([0,1500])

plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0, fontsize=4)
legend = ax.legend(fontsize = 4)
legend.get_frame().set_linewidth(0.5)
plt.tight_layout()

#%% plot width-average stress and strain
# ----------------------------------------------------------------------------
# ----- plot scatter of width-averaged stress and strain -----
# create list of arrays with strain values at each frame for boxplots

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

ax.set_xlim([0,round(all_frames_df['Exx'].max(),1)+0.5])
ax.set_ylim([0,round(all_frames_df['stress_mpa'].max(),1)+0.05])
ax.grid(True, alpha = 0.5,zorder = 0)
ax.set_xlabel('Exx')
ax.set_ylabel('Stress (Mpa)')

