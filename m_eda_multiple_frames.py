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

def convert_frame_time(x):
    return frame_time_conv[x]

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
    
def extract_mts_data(file_path, files, col_dtypes, columns):
    # import csv file
    mts_df = pd.read_csv(os.path.join(file_path,files[0]),skiprows = 5,
                         header = 1)
    # set dataframe columns
    mts_df.columns = columns
    # drop index with units
    mts_df = mts_df.drop(axis = 0, index = 0)
    # set to numeric dtype
    mts_df = mts_df.astype(col_dtypes)
    
    # filter based on trigger value and drop unneeded columns
    cam_trig_df = mts_df[mts_df['trigger'] > 0].drop(['trigger',
                                                      'cam_44','cam_43',
                                                      'trig_arduino'],
                                                     axis = 1)

    # return data at every nth frame    
    return cam_trig_df

#%% ----- MAIN SCRIPT -----
# ----------------------------------------------------------------------------
# ----- configure directories, plot colors and constants -----
# ----------------------------------------------------------------------------
# root directory
dir_root = 'Z:/Experiments/lce_tension'
# extensions to access sub-directories
batch_ext = 'lcei_001'
mts_ext = 'mts_data'
sample_ext = '007_t09_r00'
gom_ext = 'gom_results'
cmap_name = 'lajolla' # custom colormap stored in mpl_styles
frames_list_filename = batch_ext+'_'+sample_ext+'_frames_list.csv'
mts_columns = ['time', 'crosshead', 'load', 'trigger', 'cam_44', 'cam_43', 'trig_arduino']
mts_col_dtypes = {'time':'float',
              'crosshead':'float', 
              'load':'float',
              'trigger': 'int64',
              'cam_44': 'int64',
              'cam_43': 'int64',
              'trig_arduino': 'int64'}

# load single frames flag
load_multiple_frames = True
orientation = 'vertical'
# set frame range manually for plotting
frame_range = 38
mask_frame = 9
post_mask_frame = 25 # frame to compare to mask to determine if strain inc/dec
img_scale = 0.01248 # image scale (mm/pix)

# colour and edge colour arrays (hard coded to 10 or less strain bands)
ec = ['#003057', '#00313C', '#9E2A2F', '#623412','#284723',
      '#59315F', '#A45A2A', '#53565A', '#007377', '#453536']
c = ['#BDD6E6', '#8DB9CA', '#F4C3CC', '#D9B48F', '#9ABEAA',
     '#C9B1D0', '#F1C6A7', '#C8C9C7', '#88DBDF', '#C1B2B6']

# load in colormap
cm_data = np.loadtxt('Z:/Python/mpl_styles/'+cmap_name+'.txt')
custom_map = LinearSegmentedColormap.from_list('custom', cm_data)

# define custom plot style
plt.style.use('Z:/Python/mpl_styles/stg_plot_style_1.mplstyle')

# define full paths to mts and gom data
dir_xsection = os.path.join(dir_root,batch_ext,sample_ext)
dir_mts = os.path.join(dir_root,batch_ext,mts_ext,batch_ext+'_'+sample_ext)
dir_gom_results = os.path.join(dir_root,batch_ext,sample_ext,gom_ext)

num_frames = len(
    [f for f in os.listdir(dir_gom_results) if f.endswith('.pkl')]
    )

# collect mts data file
files_mts = [f for f in os.listdir(dir_mts) if f.endswith('.csv')]
#%% ----- LOAD DATA -----
# ----------------------------------------------------------------------------
# ----- load in DIC to dataframe -----
# ----------------------------------------------------------------------------
if load_multiple_frames:
    for i in range(1,42):
        print('Adding frame: '+str(i))
        save_filename = 'results_df_frame_' + '{:02d}'.format(i) + '.pkl'
        try:
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
        except:
            print('File not found or loaded succesfully.')
        
all_frames_df = all_frames_df.dropna(axis = 0)

# ----------------------------------------------------------------------------
# ----- load in MTS data and frame lists to dataframes -----
# ----------------------------------------------------------------------------

try:
    mts_df = extract_mts_data(dir_mts, files_mts, mts_col_dtypes, 
                                mts_columns)
    # reset index
    mts_df = mts_df.reset_index(drop=True)
except:
    print('No file containing mts measurements found.')
    
try:
    frames_list = pd.read_csv(os.path.join(dir_xsection,frames_list_filename))
except:
    print('Frames list was not found/loaded.')
    
mts_df = mts_df[mts_df.index.isin(frames_list['Frame list'])]
mts_df = mts_df.reset_index(drop=True)

# create dictionary of frame-time mapping from mts_df
time_mapping = mts_df.iloc[:,0].to_dict()

#%% ----- ADD FEATURES -----
# ----------------------------------------------------------------------------
# ----- add columns with scaled coordinates -----
# ----------------------------------------------------------------------------
all_frames_df['x_mm'] = all_frames_df['x_pix']*img_scale + all_frames_df['ux']
all_frames_df['y_mm'] = all_frames_df['y_pix']*img_scale + all_frames_df['uy']

# ----------------------------------------------------------------------------
# ----- calculate axial stretch from Green-Lagrange strain fields ----
# ----------------------------------------------------------------------------
all_frames_df['lambda_y'] = all_frames_df[['Eyy']].apply(lambda x: np.sqrt(2*x+1))

# ----------------------------------------------------------------------------
# -----create dataframe of points in frame used for clustering -----
# ----------------------------------------------------------------------------
mask_frame_df = all_frames_df[all_frames_df['frame'] == mask_frame]
first_frame_df = all_frames_df[all_frames_df['frame'] == 1]

# ----------------------------------------------------------------------------
# add time to dataframe based on time mapping
# ----------------------------------------------------------------------------
all_frames_df['time'] = all_frames_df['frame'].map(time_mapping)

# ----------------------------------------------------------------------------
# ----- keep points appearing in all frames in a separate df -----
# ----------------------------------------------------------------------------
# manually define last frame where all points still in FOV
end_frame = 30
last_frame_pts = all_frames_df[all_frames_df['frame'] == end_frame]

all_frames = all_frames_df.copy()

# keep only points that appear in last frame
pts_in_all_frames_df = all_frames[
all_frames.index.isin(last_frame_pts.index)]

# ----------------------------------------------------------------------------
# ----- create in-plane Poisson's ratio feature -----
# ---------------------------------------------------------------------------
try: 
    if orientation == 'vertical':
        all_frames_df['nu'] = -1*all_frames_df['Exx']/all_frames_df['Eyy']
    elif orientation == 'horizontal':
        all_frames_df['nu'] = -1*all_frames_df['Eyy']/all_frames_df['Exx']
except:
    print('Specimen orientation not recognized/specified.')

# ----------------------------------------------------------------------------
# ----- calculate temporal change in strain and incr/decr flag -----
# ----------------------------------------------------------------------------
''' 
Procedure:
1) extract first two frames with all points that appear over the full test
2) use this temp df to extract the indices of all points
3) the spacing between points is constant across frames so consider one case
4) use the row spacing between the same points to compute the strain diff
'''
# extract points in first two frames  
temp_df = pts_in_all_frames_df[pts_in_all_frames_df['frame'] == 1 ]

pt_indices = temp_df.index
# find index spacing of one point
indices_single_pt = [i for i, x in enumerate(pt_indices) if x == pt_indices[0]]
pts_period = indices_single_pt[1] - indices_single_pt[0]

pts_in_all_frames_df['dEyy_dt'] = pts_in_all_frames_df['Eyy'].diff(periods = pts_period)   

# add feature indicating if temporally increasing or decreasing beyond mask frame
# create dictionary mapping indices to value (0 or 1)
mask_df = pts_in_all_frames_df[pts_in_all_frames_df['frame'] == mask_frame]
post_mask_df = pts_in_all_frames_df[pts_in_all_frames_df['frame'] == post_mask_frame]

deyy_dt = mask_df[['Eyy']] - post_mask_df[['Eyy']]
deyy_dt_bool = (deyy_dt > 0).astype(int)

# create separate data frame with coordinates and bool mapping for strain incr.
first_frame_df['dEyy_dt_cat'] = deyy_dt_bool

#%% ----- EXPLORATORY DATA ANALYSIS -----
# ----- plot histogram and box plot of strain for each frame -----
# ----------------------------------------------------------------------------
# ----- initialize plot vars -----
# ----------------------------------------------------------------------------
n_bins = 20

num_img_x = 6
num_img_y = int(round((frame_range)/num_img_x,0))

# create list of arrays with strain values at each frame for boxplots
data_exx = []
data_eyy = []
data_exy = []

var_to_plot = 'Eyy'

# initialize row and column index counters
row, col = 0, 0
# ----------------------------------------------------------------------------
# ----- create figure -----
# ----------------------------------------------------------------------------
fig, axs = plt.subplots(num_img_y, num_img_x, sharey=True, tight_layout=True)
    
# loop through range of frames and generate histogram and box plots
plot_num = 0

for i in range(2,frame_range):
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
    
    # fix x-axis for all frames
    axs[row,col].set_xlim([0,1.05*single_frame_df['Eyy'].max()])
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
# ----------------------------------------------------------------------------
# ----- compute plot specific quantities -----
# ----------------------------------------------------------------------------
data_exx = []
data_eyy = []
data_exy = []

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
    
# ----------------------------------------------------------------------------
# ----- generate box plots -----
# ----------------------------------------------------------------------------
mpl.rcParams['lines.marker']=''
plot_boxplot_vs_frame(data_exx, frame_labels, ylabel = 'Exx')
plot_boxplot_vs_frame(data_eyy, frame_labels, ylabel = 'Eyy')
plot_boxplot_vs_frame(data_exy, frame_labels, ylabel = 'Exy')

#%% ----- Plot global stress-strain curve -----
# ----------------------------------------------------------------------------
# assign x and y to vars for brevity
x = all_frames_df.groupby('frame')['lambda_y'].mean()
y = all_frames_df.groupby('frame')['stress_mpa'].mean()
# ----------------------------------------------------------------------------
# ----- create figure -----
# ----------------------------------------------------------------------------
plot_params = {'figsize': (6,3),
               'm_size': 2,
               'linewidth': 0.5,
               'xlabel': '$\lambda_y$ (Avg. Field)',
               'ylabel': 'Eng. Stress (MPa)',
               'tight_layout': True,
               'log_x': True}

create_simple_scatter(x, y, plot_params, c, ec)

#%% ----- Plot stress-strain curves for points in view for full test -----
# ----------------------------------------------------------------------------
# assign x and y to vars for brevity
x_1 = pts_in_all_frames_df.groupby('frame')['time'].mean()
y_1 = pts_in_all_frames_df.groupby('frame')['stress_mpa'].mean()
# ----------------------------------------------------------------------------
# ----- create figure -----
# ----------------------------------------------------------------------------
plot_params = {'figsize': (6,3),
               'm_size': 2,
               'linewidth': 0.5,
               'xlabel': 'Time (s)',
               'ylabel': 'Eng. Stress (MPa)',
               'tight_layout': True,
               'log_x': True}

create_simple_scatter(x_1, y_1, plot_params, c, ec)

#%% ----- filter data into strain ranges based on selected frame -----
# ----------------------------------------------------------------------------
# ----- initialize plot vars -----
# ----------------------------------------------------------------------------
num_category_bands = 6
y_var = 'Eyy'
x_var = 'time'
cat_var = 'Eyy'
num_samples_to_plot = 8000

num_img_x = 3
num_img_y = int((num_category_bands)/num_img_x)
# ----------------------------------------------------------------------------
# ----- compute plot specific quantities -----
# ----------------------------------------------------------------------------
# calculate strain range bounds
max_category_band = round(mask_frame_df[cat_var].quantile(0.98),1)
min_category_band = round(mask_frame_df[cat_var].min(),1)

category_ranges = np.linspace(min_category_band, max_category_band, num_category_bands)

# compute average Green-Lagrange strain for points in all frames
avg_strain = pts_in_all_frames_df.groupby(x_var)['Eyy'].mean()
# ----------------------------------------------------------------------------
# ----- create figure -----
# ----------------------------------------------------------------------------
fig, axs = plt.subplots(num_img_y, num_img_x, figsize=(5,5),
                        sharey=True)#, tight_layout=True)

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
        
        axs[row,col].plot(
            avg_strain, linewidth = 0.5, linestyle = '-', c = 'k',  
            alpha = 1.0)
    
        axs[row,col].set_ylim([0, round(
            all_frames_df[y_var].quantile(0.995),1
            )
            +0.4
            ])
        
        #axs[row,col].set_ylim([0, 0.4])

        try:
            if x_var == 'time':
                axs[row,col].set_xscale('log')
        except:
            axs[row,col].set_xscale('linear')
        #axs[row,col].set_xlim([0,len(all_frames_df[dependent_var].unique())+2])
        axs[row,col].set_xlim([0,round(
            all_frames_df[x_var].quantile(0.995),1
            )
            +0.5
            ])
        axs[row,col].set_ylabel(y_var)
        axs[row,col].set_xlabel(x_var)
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
    
        _, max_ylim = plt.ylim()
        if x_var == 'time':
            # add line showning mean of field at each frame
            axs[row,col].axvline(time_mapping[mask_frame], linestyle='dashed', 
                     linewidth=0.4, marker = '')
            axs[row,col].text(time_mapping[mask_frame]*1.1, max_ylim*0.8,
                  'Mask frame: {:.0f}'.format(time_mapping[mask_frame])+ ' s',
                  fontsize = 4)
        else:
            # add line showning mean of field at each frame
            axs[row,col].axvline(mask_frame, linestyle='dashed', 
                             linewidth=0.4, marker = '')
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
            axs[row,col].set_ylim([min_category_band-0.1,max_category_band])
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
# ----------------------------------------------------------------------------
# ----- initialize plot vars -----
# ----------------------------------------------------------------------------
cat_var = 'Eyy'
plot_var = 'Eyy'

# ----------------------------------------------------------------------------
# ----- create figure -----
# ----------------------------------------------------------------------------
f = plt.figure(figsize = (3,3))
ax = f.add_subplot(1,1,1)

for i in range(1,num_category_bands+1):
    if i == num_category_bands:
        category_band_df = mask_frame_df[
            mask_frame_df[cat_var] >= category_ranges[i-1]]   
    else:
        category_band_df = mask_frame_df[(
            mask_frame_df[cat_var] >= category_ranges[i-1]
            ) & (
            mask_frame_df[cat_var] < category_ranges[i]
            )]
            
    # find all points within that strain range
    agg_category_band_df = all_frames_df[
    all_frames_df.index.isin(category_band_df.index)]
    
    if agg_category_band_df.shape[0] > 4000:
        category_band_sample = agg_category_band_df.sample(n = 4000, 
                                                       random_state = 1)
    else:
        category_band_sample = agg_category_band_df.copy()
    
    # add to plot
    ax.scatter(
        category_band_sample[plot_var], 
        category_band_sample['stress_mpa'],
        s = 2, 
        c = c[i-1],
        edgecolors = ec[i-1], 
        alpha = 0.4,
        linewidths = 0.5, 
        label = 'strain_band: '+str(i)
        )
    
    ax.set_xlim([0,round(all_frames_df[plot_var].max(),1)+0.5])
    #ax.set_xlim([0,3.6])
    ax.set_ylim([0,round(all_frames_df['stress_mpa'].max(),1)+0.02])
    ax.grid(True, alpha = 0.5,zorder = 0)
    ax.set_xlabel('Eyy')
    ax.set_ylabel('Stress (Mpa)')
plt.legend(loc='upper left', fontsize = 4)

legend = ax.legend(fontsize = 4)
legend.get_frame().set_linewidth(0.5)
plt.tight_layout()

#%% ----- OVERLAY STRAIN BAND LOCATIONS ON UNDEFORMED SAMPLE -----
# ----------------------------------------------------------------------------
# ---- initialize plot vars -----
# ----------------------------------------------------------------------------
cat_var = 'Eyy'

# ----------------------------------------------------------------------------
# ---- create figure -----
# ----------------------------------------------------------------------------
f = plt.figure(figsize = (1.62,4))
ax = f.add_subplot(1,1,1)
for i in range(0,num_category_bands+1):
    if i == num_category_bands:
            category_band_df = mask_frame_df[
                mask_frame_df[cat_var] >= category_ranges[i-1]]   
    else:
        category_band_df = mask_frame_df[(
            mask_frame_df[cat_var] >= category_ranges[i-1]
            ) & (
            mask_frame_df[cat_var] < category_ranges[i]
            )]
    
    incl_category_band_df = all_frames_df[
        all_frames_df.index.isin(category_band_df.index)]
    
    # plot on one figure
    if i == 0:
        ax.scatter(
            all_frames_df[all_frames_df['frame'] == 1][['x_pix']]*img_scale, 
            all_frames_df[all_frames_df['frame'] == 1][['y_pix']]*img_scale, 
            s = 1, c = '#D0D3D4', edgecolors = '#D0D3D4', 
            alpha = 0.3, linewidths = 0, zorder = 0, label = 'all_other_points'
            )
    else:
        ax.scatter(
            incl_category_band_df[incl_category_band_df['frame'] == 1][['x_pix']]*img_scale, 
            incl_category_band_df[incl_category_band_df['frame'] == 1][['y_pix']]*img_scale, 
            s = 1, c = c[i-1], edgecolors = ec[i-1], alpha = 1, 
            linewidths = 0, zorder = i, label = 'Range: '+str(i)
            )

# ----- customize plot - add grid, labels, and legend -----
ax.grid(True, alpha = 0.5)
ax.set_xlabel('x (mm)')
ax.set_ylabel('y (mm)')
ax.set_xlim([12,19])

# shrink current axes box to place legend overhead with axes labels
# Shrink current axis's height by 10% on the bottom
'''
box = ax.get_position()
ax.set_position(
    [box.x0, box.y0 + box.height * 0.25,
     box.width, box.height * 0.5]
    )

legend = ax.legend(
    fontsize = 3, loc='upper right', ncol=1, 
    bbox_to_anchor=(0, 0.1)
    )
legend.get_frame().set_linewidth(0.5)
'''
plt.tight_layout()
plt.show()

#%% ----- OVERLAY INCREASING/DECREASING STRAIN LOCATIONS ON UNDEFORMED SAMPLE -----
# ----------------------------------------------------------------------------
# ---- initialize plot vars -----
# ----------------------------------------------------------------------------
cat_var = 'dEyy_dt_cat'
mask_frame_pts_all_frames_df = pts_in_all_frames_df[pts_in_all_frames_df['frame'] == mask_frame]

# ----------------------------------------------------------------------------
# ---- create figure -----
# ----------------------------------------------------------------------------
f = plt.figure(figsize = (1.62,4))
ax = f.add_subplot(1,1,1)
# plot all points in reference config

ax.scatter(
    all_frames_df[all_frames_df['frame'] == 1][['x_pix']]*img_scale, 
    all_frames_df[all_frames_df['frame'] == 1][['y_pix']]*img_scale, 
    s = 1, c = '#D0D3D4', edgecolors = '#D0D3D4', 
    alpha = 0.3, linewidths = 0, zorder = -1, label = 'all_other_points'
    )

        
for i in [0,1]:
    category_band_df = first_frame_df[
               first_frame_df[cat_var] == i]
    
    # plot on one figure
    ax.scatter(
        category_band_df[['x_pix']]*img_scale, 
        category_band_df[['y_pix']]*img_scale, 
        s = 1, c = c[i], edgecolors = ec[i], alpha = 1, 
        linewidths = 0, zorder = i, label = 'Range: '+str(i)
        )

# ----- customize plot - add grid, labels, and legend -----
ax.grid(True, alpha = 0.5)
ax.set_xlabel('x (mm)')
ax.set_ylabel('y (mm)')
ax.set_xlim([12,19])

# shrink current axes box to place legend overhead with axes labels
# Shrink current axis's height by 10% on the bottom

box = ax.get_position()
ax.set_position(
    [box.x0, box.y0 + box.height * 0.25,
     box.width, box.height * 0.5]
    )

legend = ax.legend(
    fontsize = 3, loc='upper right', ncol=1, 
    bbox_to_anchor=(0, 0.1)
    )
legend.get_frame().set_linewidth(0.5)

plt.tight_layout()
plt.show()

#%% ----- PLOT WIDTH-AVERAGED STRESS VS STRAIN -----
# ----------------------------------------------------------------------------
# ----- create figure -----
# ----------------------------------------------------------------------------
f = plt.figure(figsize = (3,3))
ax = f.add_subplot(1,1,1)
for i in range(1,frame_range):
    df = all_frames_df[all_frames_df['frame'] == i]

    # group by x pix and take average of stress and strain
    df_xs_group = df.groupby('x_pix').mean()

    ax.scatter(df_xs_group[['Eyy']], 
               df_xs_group[['stress_mpa']], 
               s = 1, 
               c = '#BDD6E6', 
               edgecolors = '#003057', 
               alpha = 0.4,
               linewidths = 0.5)

#ax.set_xlim([0,round(all_frames_df['Eyy'].max(),1)-1])
ax.set_xlim([0,3])
ax.set_ylim([0,round(all_frames_df['stress_mpa'].max(),1)+0.02])
ax.grid(True, alpha = 0.5,zorder = 0)
ax.set_xlabel('Eyy')
ax.set_ylabel('Stress (Mpa)')

#%% ----- EXX VS EYY (CHECK COMPRESSIBILITY) -----
# ----------------------------------------------------------------------------
# ----- Initialize plot vars -----
# ----------------------------------------------------------------------------
num_category_bands = 6
y_var = 'Exx'
x_var = 'Eyy'
cat_var = 'Eyy'
num_samples_to_plot = 4000

x_fit = np.linspace(0,3,500) # Eyy
y_fit_1 = 0.5*(1/(2*x_fit+1) - 1)
y_fit_2 = 0.5*(1/np.sqrt(1+2*x_fit) - 1)

num_img_x = 1
num_img_y = 1

# ----------------------------------------------------------------------------
# ----- compute plot specific quantities -----
# ----------------------------------------------------------------------------
# calculate category range bounds
max_category_band = round(mask_frame_df[cat_var].quantile(0.98),1)
min_category_band = round(mask_frame_df[cat_var].min(),1)

category_ranges = np.linspace(
    min_category_band, max_category_band, num_category_bands
    )

# ----------------------------------------------------------------------------
# ---- create figure -----
# ----------------------------------------------------------------------------
fig, axs = plt.subplots(num_img_x, num_img_y, figsize=(3,3),
                        sharey=True, tight_layout=True)

for i in range(1,num_category_bands+1):

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
    axs.set_ylim([round(
        all_frames_df[y_var].quantile(0.001),2
        )*1.2, round(
        all_frames_df[y_var].quantile(0.995),2
        )*1.2
        ])

    axs.set_xlim([0,round(
        all_frames_df[x_var].quantile(0.995),1
        )*1.2
        ])
    #axs.set_ylim([-0.45,0])
    #axs.set_xlim([0,1.8])
    axs.set_ylabel(y_var)
    axs.set_xlabel(x_var)
    axs.grid(True, alpha = 0.5,zorder = 0)  

# compare against incompressible relationship    
axs.plot(
    x_fit, y_fit_1, linewidth = 0.5, linestyle = '--', c = 'k',  
    alpha = 1.0, label = '$\lambda_x = \lambda_y^{-1}$')
axs.plot(
    x_fit, y_fit_2, linewidth = 0.5, linestyle = '-', c = 'k', 
    alpha = 1.0, label = '$\lambda_x = \lambda_y^{-1/2}$')

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

#%% ----- Plot 3D SCATTER (STRAIN COMPONENTS VS ROTATION) ---
'''
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
'''
