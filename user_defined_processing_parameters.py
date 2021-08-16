# -*- coding: utf-8 -*-
"""
USER-DEFINED ANALYSIS AND PLOT PARAMETERS
"""
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================================
# -------------- USER INPUT ----------------
# ============================================================================
# ---------- Data directories ----------
# ----------------------------------------------------------------------------
dir_root = 'Z:/Experiments/lce_tension'
# extensions to access sub-directories
batch_ext = 'lcei_003'
mts_ext = 'mts_data'
sample_ext = '001_t05_r00'
gom_ext = 'gom_results'
frame_map_ext = 'frame_time_mapping'

# ---------- Columns and data types of mts data ---------- 
mts_columns = ['time', 'crosshead', 'load', 'trigger', 
               'cam_44', 'cam_43', 'trig_arduino']
mts_col_dtypes = {'time':'float', 'crosshead':'float', 'load':'float',
              'trigger': 'int64', 'cam_44': 'int64', 'cam_43': 'int64',
              'trig_arduino': 'int64'}
# ----------------------------------------------------------------------------
# ---------- Processing options ----------
# ----------------------------------------------------------------------------
load_multiple_frames = False # True if one wants to load all frames into memory simultaneously
orientation = 'vertical' # Orientation of pulling axis w.r.t camera
frame_min = 1 # min frame to plot
frame_max = 36 # max frame to consider
frame_range = frame_max - frame_min
#
frame_rel_min = 5 # start frame for computing relative change between frames
end_frame = 36 # manually define last frame of test/where all points still in FOV
mask_frame = 5 # frame to use to mask points for clustering
peak_frame_index = 5 # frame where load is max for normalizing stress/strain relax rates
img_scale = 0.02724 # image scale (mm/pix)
# ----------------------------------------------------------------------------
# -------- Plotting ---------
# ----------------------------------------------------------------------------
# -------- Specify which plots to generate ----------
# NOTE: 'all plot options' only for reference and not called in main script
plt_to_generate = [
                   'global_stress_strain',
                   'var_clusters_vs_time_subplots',
                   'norm_stress_strain_rates_vs_time',
                   'var_vs_time_clusters_same_axis'
                   ]

all_plot_options = [
                    'boxplot',
                   'histogram',
                   'global_stress_strain',
                   'var_clusters_vs_time_subplots',
                   'overlay_pts_on_sample_var',
                   'overlay_pts_on_sample_relative',
                   'compressibility_check',
                   'norm_stress_strain_rates_vs_time',
                   'var_vs_time_clusters_same_axis'
                   ]

# ----- Set general plot parameters imported to all plot functions -----
# NOTE: can override for select plots using additional plot param dictionaries
#       which are defined below
plt_params_general = {
    'fontsize': 5,
    'fontsize2': 4,
    'linewidth': 0.5,
    'linewidth2': 0.3,
    'linestyle': '-',
    'linestyle2': '--',
    'm_size': 2,
    'marker': 'o',
    'marker2': '^',
    'm_alpha': 0.4,
    'grid_alpha': 0.5,
    'axes_scaled': False,
    'log_x': True,
    'legend_fontsize': 4,
    'legend_linewidth': 0.5,
    'legend_loc': 'upper center',
    'legend_m_size': 7,
    'legend_fancybox': False,
    'legend_ncol': 1,
    }

# ---------- Colour palettes used for 7, 2 and 1 series plots ----------
# colour and edge colour arrays (hard coded to 7 or less strain bands)
ec = ['#917265', '#896657', '#996f71', '#805a66', '#453941',
      '#564e5c', '#32303a']
c = ['#d1c3bd', '#ccb7ae', '#b99c9d', '#a6808c', '#8b7382',
     '#706677', '#565264']
ec2 = ['#917265', '#564e5c']
c2 = ['#d1c3bd', '#706677']
ec1 = ['#564e5c']
c1 = ['#706677']

# define custom plot style
plt.style.use('Z:/Python/mpl_styles/stg_plot_style_1.mplstyle')

# ----------------------------------------------------------
# ----- DEFINE PLOT/ANALYSIS PARAMETERS FOR SPECIFIC PLOTS
# ----------------------------------------------------------
# ----------------------------------------------------------------------------
# ----- HISTOGRAM -----
# ----------------------------------------------------------------------------
subplot_cols = 6
subplot_dims = [int(np.floor((frame_range-1)/subplot_cols)+1), subplot_cols]

plt_params_histogram = {
                'figsize': (4,4),
                'xlabel': 'Frame number',
                'ylabel': 'Eyy',
                'n_bins': 20,
                'subplot_dims': subplot_dims,
                'ec': ec1,
                'c': c1
                }

anlys_params_histogram = {'plot_var': 'Eyy'}
# ----------------------------------------------------------------------------
# ----- BOXPLOT -----
# ----------------------------------------------------------------------------
plt_params_boxplot = {
                'figsize': (8,2),
                'xlabel': 'Frame number',
                'ylabel': 'Eyy',
                'showfliers': False,
                'xlims': [frame_min-1, frame_max+1]
                }

anlys_params_boxplot = {'plot_var': 'Eyy'}
# ----------------------------------------------------------------------------
# ----- GLOBAL_STRESS_STRAIN -----
# ----------------------------------------------------------------------------
plt_params_glob_ss = {
                    'figsize': (6,2),
                    'xlabel': 'Time (s)',
                    'ylabel': 'Eng. Stress (MPa)',
                    'ec': ec1,
                    'c': c1,
                    'label': 'lcei_003_001_t03_r04',
                    'legend_loc': 'upper right'
                    }

anlys_params_glob_ss = {
                    'x': 'time',
                    'y': 'stress_mpa'
                    }
# ----------------------------------------------------------------------------
# ----- VAR_CLUSTERS_VS_TIME_SUBPLOTS -----
# ----------------------------------------------------------------------------
num_categories = 6
subplot_cols = 3
subplot_dims = [int(np.floor((num_categories-1)/subplot_cols)+1), subplot_cols]
        
plt_params_var_clusters_subplots = {
                    'figsize': (4,4),
                    'xlabel': 'Time (s)',
                    'ylabel': 'Eng. Stress (MPa)',
                    'subplot_dims': subplot_dims,
                    'ec': ec,
                    'c': c
                    }
  
anlys_params_var_clusters_subplots = {
                    'x_var': 'time',
                    'y_var': 'Eyy',
                    'cat_var': 'Eyy',
                    'num_categories': num_categories,
                    'samples': 8000,
                    'mask_frame': mask_frame,
                    }
# ----------------------------------------------------------------------------
# ----- OVERLAY_PTS_ON_SAMPLE_VAR -----
# ----------------------------------------------------------------------------
plt_params_pts_overlay_var = {
                    'figsize': (2,4),
                    'linewidth': 0,
                    'xlabel': 'x (mm)',
                    'ylabel': 'y (mm)',
                    'ref_c': '#D0D3D4',
                    'ref_ec': '#D0D3D4',
                    'axis_scaled': True,
                    'c': c,
                    'legend_ncol': 3
                    }

anlys_params_pts_overlay_var = {
                    'var_interest': 'Eyy',
                    'cat_var': 'Eyy',
                    'num_categories': 6
    }
# ----------------------------------------------------------------------------
# ----- OVERLAY_PTS_ON_SAMPLE_RELATIVE -----
# ----------------------------------------------------------------------------
plt_params_pts_overlay_relative = {'figsize': (2,4),
                      'linewidth': 0,
                      'xlabel': 'x (mm)',
                      'ylabel': 'y (mm)',
                      'ref_c': '#D0D3D4',
                      'ref_ec': '#D0D3D4',
                      'axis_scaled': True,
                      'c': c,
                      'legend_ncol': 1
                      }

anlys_params_pts_overlay_relative = {
                    'var_interest': 'Eyy',
                    'cat_var': 'dEyy/dt',
                    'num_categories': 2
                    }
# ----------------------------------------------------------------------------
# ----- COMPRESSIBILITY_CHECK -----
# ----------------------------------------------------------------------------
plt_params_comp_check = {
                    'figsize': (3,3),
                    'xlabel': 'Eyy',
                    'ylabel': 'Exx',
                    'y_fit_1_label': '$\lambda_x = \lambda_y^{-1}$',
                    'y_fit_2_label': '$\lambda_x = \lambda_y^{-1/2}$',
                    'ec': ec,
                    'c': c,
                    'legend_loc': 'lower right'
                    }

# generate theoretical compressibility relationships
x_max = 3
num_pts = 500
# ---- compute (DO NOT MODIFY) -----
x_fit = np.linspace(0,x_max,num_pts) # Eyy
y_fit_1 = 0.5*(1/(2*x_fit+1) - 1)
y_fit_2 = 0.5*(1/np.sqrt(1+2*x_fit) - 1)
# ------------------------------------------
        
# define analysis parameters dictionary
anlys_params_comp_check = {
                    'x_var': 'Eyy',
                    'y_var': 'Exx',
                    'cat_var': 'Eyy',
                    'num_categories': num_categories,
                    'samples': 8000,
                    'mask_frame': mask_frame,
                    'x_fit': x_fit,
                    'y_fit_1': y_fit_1,
                    'y_fit_2': y_fit_2
                    }
# ----------------------------------------------------------------------------
# ----- VAR_VS_TIME_CLUSTERS_SAME_AXIS -----
# ----------------------------------------------------------------------------
plt_params_var_vs_time_clusters_sa = {
                    'figsize': (3,3),
                    'xlabel': 'time',
                    'ylabel': 'Eyy',
                    'labels': ['dEyy/dt < 0', 'dEyy/dt >= 0'],
                    'm_alpha': 1,
                    'legend_loc': 'upper right',
                    'ec': ec2,
                    'c': c2
                    }

anlys_params_var_vs_time_clusters_sa = {
                    'x_var': 'time',
                    'y_var': 'Eyy', 
                    'cat_var': 'dEyy/dt',
                    'samples': 8000,
                    'num_categories': 2}
# ----------------------------------------------------------------------------
# ----- NORM_STRESS_STRAIN_RATES_VS_TIME -----
# ----------------------------------------------------------------------------
plt_params_norm_ss_rates_vs_time = {
                'figsize': (3,3),
                'xlabel': 'time',
                'ylabel': 'norm(dX/dt)',
                'labels_y1': ['Eyy [Eyy < 0]', 'Eyy [Eyy >= 0]'],
                'labels_y2': ['$\sigma$ [Eyy < 0]', '$\sigma$ [Eyy >= 0]'],
                'log_y': True,
                'm_alpha': 1,
                'ec': ec2,
                'c': c2,
                'legend_loc': 'upper right'
                }

anlys_params_norm_ss_rates_vs_time = {
                    'x_var': 'time',
                    'y_var': 'Eyy',
                    'y_var_2': 'stress_mpa',
                    'cat_var': 'Eyy',
                    'num_categories': 2,
                    'normalize_y': False,
                    'peak_frame_index': peak_frame_index}

# ============================================================================
# -------------- CALCULATED INPUTS (DO NOT MODIFY) ----------------
# ============================================================================
# define full paths to mts and gom data
dir_frame_map = os.path.join(dir_root,batch_ext,mts_ext,batch_ext+'_'+sample_ext,frame_map_ext)
dir_mts = os.path.join(dir_root,batch_ext,mts_ext,batch_ext+'_'+sample_ext)
dir_results = os.path.join(dir_root,batch_ext,sample_ext,gom_ext)

frame_map_filename = batch_ext+'_'+sample_ext+'_frame_time_mapping.csv'

results_files = [f for f in os.listdir(dir_results) if f.endswith('.pkl')]
num_frames = len(results_files)

# ---------- plot ranges ----------
plt_frame_range = [frame_min, frame_max] # range of frames to plot

# ----- combine all plot specific parameter dicts with general parameters -----
plt_params_histogram = {**plt_params_general, **plt_params_histogram}
plt_params_boxplot = {**plt_params_general, **plt_params_boxplot}
plt_params_glob_ss = {**plt_params_general, **plt_params_glob_ss}
plt_params_var_clusters_subplots = {**plt_params_general,
                                    **plt_params_var_clusters_subplots}
plt_params_comp_check = {**plt_params_general,
                                    **plt_params_comp_check}
plt_params_pts_overlay_var = {**plt_params_general,
                                    **plt_params_pts_overlay_var}
plt_params_pts_overlay_relative = {**plt_params_general,
                                    **plt_params_pts_overlay_relative}
plt_params_var_vs_time_clusters_sa = {**plt_params_general,
                                    **plt_params_var_vs_time_clusters_sa}
plt_params_norm_ss_rates_vs_time = {**plt_params_general,
                                    **plt_params_norm_ss_rates_vs_time}