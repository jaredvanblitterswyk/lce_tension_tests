# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 17:55:43 2021

@author: jcv
"""
import os
import sys
import csv
import pandas as pd
import numpy as np
from numpy import linalg as LA
from scipy import linalg
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.font_manager as font_manager
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import interp2d, griddata
import matplotlib.tri as tri
import math as m

# -- TO DO: would need to reformat m_process_dic_fields_btb_imaging
# currently not setup up properly for importing functions as a package

#sys.path.insert(0, 'Z:/Python/image_processing_shear_tests/')
#from m_process_dic_fields_btb_imaging import calculateEij

def compute_R(F):
    # compute rotation matrix iteratively using technique presented in:
    # http://www.continuummechanics.org/polardecomposition.html
    
    # compute transpose of inverse
    Fti = LA.inv(F.T)
    
    # compute norm of matrix - iterative way of computing R instead of matrix 
    # sqrt calc
    Ao = np.zeros((F.shape[0],F.shape[1]))
    An = 0.5*(F + Fti)
    # define the difference between matrices using the normalized distance
    res = LA.norm(An - Ao)
    
    threshold = 0.01
    while res > threshold:
        Ao = An
        Ati = LA.inv(Ao.T)
        An = 0.5*(Ao + Ati)
        res = LA.norm(An - Ao)
        
    return An
    
def calculateEijRot(disp, strain_labels, spacing):
    # pull out displacement fields
    disp_x = disp.get('ux')
    disp_y = disp.get('uy')
    
    # pull out measurement point spacing
    dx = spacing[0]
    dy = spacing[1]
    
    #compute displacement gradient fields
    du_dy, du_dx = np.gradient(disp_x)
    dv_dy, dv_dx = np.gradient(disp_y)
    
    # gradient uses a unitless spacing of one, convert to mm
    du_dy = -1*du_dy/dy; dv_dy = -1*dv_dy/dy
    du_dx = du_dx/dx; dv_dx = dv_dx/dx
    
    row, col = disp_x.shape
    
    # initialize empty variables for strain and rotation fields
    Eij = {}
    for component in strain_labels:
        Eij[component] = np.zeros((row,col))
    
    Rij = np.zeros((row,col))
    Reig = np.zeros((row,col))
    stretch_p1 = np.zeros((row,col))
      
    I = np.matrix(([1,0],[0,1]))
    
    for i in range(0,row):
        for j in range(0,col):
            # calculate deformation tensor
            eij = np.zeros((2,2))
            Fij = np.zeros((2,2))
            
            Fij[0,0] = du_dx[i,j]
            Fij[0,1] = du_dy[i,j]
            Fij[1,0] = dv_dx[i,j]
            Fij[1,1] = dv_dy[i,j]
            
            Fij += I
            
            C = np.matmul(np.transpose(Fij),Fij)
            eij = 0.5*(C-I)
            #C = np.matmul(np.transpose(Fij),Fij)

            # calculate Lagrange strains
            Eij['Exx'][i,j] = eij[0,0]
            Eij['Eyy'][i,j] = eij[1,1]
            Eij['Exy'][i,j] = eij[0,1]
            
            R = compute_R(Fij)
                        
            # compute rotation matrix - limit to 1, -1
            if R[0,0] > 1:
                R[0,0] = 1
            elif R[0,0] < -1:
                R[0,0] = -1
                
            Rij[i,j] = np.arccos(R[0,0])*180/m.pi
            
            if np.isnan(np.sum(C)):
                stretch_p1[i,j] = np.nan
                Reig[i,j] = np.nan
            else:
                # compute eigenvalues and eigenvectors of stretch tensor
                eig_vec_val = linalg.eig(C, left=False, right=True)
                eigen = eig_vec_val[1][:,0]
                stretch_p1[i,j] = eig_vec_val[0][0]
                
                # compute magnitude of principal eigenvector
                Reig[i,j] = np.arccos(np.real(eigen[0]))*180/m.pi
            
    return Eij, Rij, Reig, stretch_p1

def mask_interp_region(triang, df, mask_side_len = 0.2):
    triangle_mask = []
    tr = triang.triangles
    for row in range(tr.shape[0]):

        x1, x2, x3 = df['x'][tr[row,0]], df['x'][tr[row,1]], df['x'][tr[row,2]]
        y1, y2, y3 = df['y'][tr[row,0]], df['y'][tr[row,1]], df['y'][tr[row,2]]
        
        # calculate side lengths of each triangle
        a = np.sqrt((x1-x2)**2 + (y1-y2)**2)
        b = np.sqrt((x1-x3)**2 + (y1-y3)**2)
        c = np.sqrt((x2-x3)**2 + (y2-y3)**2)
        
        # check square of side lengths - if all less than radius, keep in mask
        if (a < mask_side_len and b < mask_side_len and c < mask_side_len):
            triangle_mask.append(0)
        else: 
            triangle_mask.append(1)
            
    # mask zero area triangles
    xy = np.dstack((triang.x[triang.triangles], triang.y[triang.triangles]))  # shape (ntri,3,2)
    twice_area = np.cross(xy[:,1,:] - xy[:,0,:], xy[:,2,:] - xy[:,0,:])  # shape (ntri)
    area_mask = twice_area < 1e-3  # shape (ntri)
    area_mask = area_mask.astype(int)
               
    mask = np.maximum.reduce([np.array(triangle_mask),area_mask])
    #mask = triangle_mask
            
    return mask, area_mask
    
def extract_load_at_images(file_path, files, col_dtypes, columns, nth_frames):
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
    cam_trig_df = mts_df[mts_df['trigger'] > 0].drop(['time','trigger',
                                                      'cam_44','cam_43',
                                                      'trig_arduino'],
                                                     axis = 1)

    # return data at every nth frame    
    return cam_trig_df.iloc[::nth_frames,:]

def interp_and_calc_strains(file, mask_side_length, spacing, disp_labels, strain_labels):
    # load in data
    df = pd.read_csv(os.path.join(dir_gom_results,file), skiprows = 5)
    
    df = df.dropna(axis = 0)
    df = df.reset_index(drop=True)
    # transform GOM coords back to reference configuration
    X = df['x']+(Nx/2)*img_scale - df['displacement_x']
    Y = df['y']+(Ny/2)*img_scale - df['displacement_y']
      
    # define triangulation interpolation on x and y coordinates frm GOM
    # in reference coordinates
    triang = tri.Triangulation(X, Y)
    # get mask
    triangle_mask, area_mask = mask_interp_region(triang, df, mask_side_length)
    # apply mask
    triang.set_mask(np.array(triangle_mask) > 0)
    
    switcher = {'ux': 'displacement_x',
                'uy': 'displacement_y',
                'uz': 'displacement_z'}
    
    disp = {}
        
    # interpolate displacements to pixel coordinates
    for component in disp_labels:
    
        interpolator = tri.LinearTriInterpolator(
            triang, df[switcher.get(component)]
            )
        
        disp[component] = interpolator(xx, yy)
    
    # calculate strains and rotations from deformation gradient            
    Eij, Rij, Reig, stretch_p1 = calculateEijRot(disp, strain_labels, spacing)
    
    return disp, Eij, Rij, Reig, stretch_p1, area_mask, triangle_mask

def calc_xsection(dir_xsection, filename, img_scale, orientation):  
    # read in coordinates from imageJ analysis stored in csv
    section_df = pd.read_csv(os.path.join(dir_xsection,filename), 
                             encoding='UTF-8')
    # drop unnecessary values
    section_df.drop(columns = 'Value', inplace = True)
    if orientation == 'horizontal':
        max_c = section_df.groupby(['X']).max()
        min_c = section_df.groupby(['X']).min()
    else:
        max_c = section_df.groupby(['Y']).max()
        min_c = section_df.groupby(['Y']).min()
    
    width = max_c - min_c
    width_mm = width*img_scale   

    return width_mm
    
#%% ---- MAIN SCRIPT ----
# ----- configure directories -----
# root directory
dir_root = 'Z:/Experiments/lce_tension'
dir_root_local = 'C:/Users/jcv/Documents'
# extensions to access sub-directories
batch_ext = 'lcei_001'
mts_ext = 'mts_data'
sample_ext = '007_t09_r00'
gom_ext = 'gom_results'

# define full paths to mts and gom data
try:
    dir_xsection = os.path.join(dir_root,batch_ext,sample_ext)
    dir_mts = os.path.join(dir_root,batch_ext,mts_ext,batch_ext+'_'+sample_ext)
    dir_gom_results = os.path.join(dir_root,batch_ext,sample_ext,gom_ext)
except:
    print('One of setup directories does not exist.')

# ----- define constants -----
spec_id = batch_ext+'_'+sample_ext # full specimen id
Nx, Ny = 2448, 2048 # pixel resolution in x, y axis
img_scale = 0.01248 # mm/pix
t = 1.6 # thickness of sample [mm]
orientation = 'vertical'
cmap_name = 'lajolla' # custom colormap stored in mpl_styles
xsection_filename = batch_ext+'_'+sample_ext+'_section_coords.csv'
mask_side_length = 1.2 # max side length of triangles in DeLauny triangulation
cbar_levels = 25 # colorbar levels
nth_frames = 5 # sub sampling images where correlation data is available
mts_columns = ['time', 'crosshead', 'load', 'trigger', 'cam_44', 'cam_43', 'trig_arduino']
mts_col_dtypes = {'time':'float',
              'crosshead':'float', 
              'load':'float',
              'trigger': 'int64',
              'cam_44': 'int64',
              'cam_43': 'int64',
              'trig_arduino': 'int64'}

# ---- calculate quantities and collect files to process ---
# collect all csv files in gom results directory
files_gom = [f for f in os.listdir(dir_gom_results) if f.endswith('.csv')]
# collect mts data file
files_mts = [f for f in os.listdir(dir_mts) if f.endswith('.csv')]

# create vector of x dimensions
x_vec = np.linspace(0,Nx-1,Nx) # vector - original x coords (pix)
y_vec = np.linspace(Ny-1,0,Ny) # vector - original y coords (pix)

xx_pix,yy_pix = np.meshgrid(x_vec,y_vec) # matrix of x and y coordinates (pix)
xx,yy = xx_pix*img_scale, yy_pix*img_scale # matrix of x and y coordinates (mm)

# calculate scaled spacing between points in field
dx, dy = img_scale, img_scale

# calculate width of specimen at each pixel location
try:
    width_mm = calc_xsection(
        dir_xsection, xsection_filename, img_scale, orientation
        )
except: 
    print('No file containing cross-section coordinates found.')

try:
    mts_df = extract_load_at_images(dir_mts, files_mts, mts_col_dtypes, 
                                mts_columns, nth_frames)
    # reset index
    mts_df = mts_df.reset_index(drop=True)
except:
    print('No file containing mts measurements found.')

# assemble results dataframe
coords_df = pd.DataFrame()
coords_df['x_pix'] = np.reshape(xx_pix,(Nx*Ny,))
coords_df['y_pix'] = np.reshape(yy_pix,(Nx*Ny,))

def plot_rotation_field(x,y,Rij, frame_no):
    plt.figure(figsize = (2,3))
    plt.scatter(x, y, s = 1, c = Rij)
    plt.colorbar()
    plt.title('Frame no.: '+str(frame_no))
    plt.xlabel('x (pix)')
    plt.ylabel('y (pix)')
#%%
# ---- run processing -----  
disp_labels = ['ux', 'uy', 'uz']
strain_labels = ['Exx', 'Eyy', 'Exy']
 
for i in range(len(files_gom)-1,len(files_gom)):
    # extract frame number and display
    frame_no = files_gom[i][28:-10] # lcei_001_006_t02_r00
    #frame_no = files_gom[i][35:-10] 
    print('Processing frame:' + str(frame_no))
    
    spacing = [dx, dy]
    
    # compute interpolated strains and displacements in reference coordinates
    disp, Eij, Rij, Reig, stretch_p1, area_mask, triangle_mask = interp_and_calc_strains(
        files_gom[i], mask_side_length, spacing, disp_labels, strain_labels
        )
    
    # assemble results in data frame
    outputs_df = pd.DataFrame()
    
    for component in disp_labels:
        outputs_df[component] = np.reshape(disp.get(component),(Nx*Ny,))
    
    for component in strain_labels:
        outputs_df[component] = np.reshape(Eij.get(component),(Nx*Ny,))

    outputs_df['R'] = np.reshape(Rij,(Nx*Ny,))
    outputs_df['Reig'] = np.reshape(Reig,(Nx*Ny,))
    outputs_df['lambda1'] = np.reshape(stretch_p1,(Nx*Ny,))
    
    # ----- compile results into dataframe -----
    # concatenate to create one output dataframe
    results_df = pd.concat([coords_df,outputs_df],axis = 1, join = 'inner')
    # drop points with nans
    results_df = results_df.dropna(axis=0, how = 'any')
    # assign cross-section width based on pixel location
    results_df['width_mm'] = results_df['x_pix']
    try:
        results_df['width_mm'] = results_df['width_mm'].apply(lambda x: width_mm.loc[x][0] if x in width_mm.index else np.nan)
        # assign cross-section area
        results_df['area_mm2'] = results_df['width_mm'].apply(lambda x: x*t)
        # assign width-averaged stress
        results_df['stress_mpa'] = mts_df.iloc[int(frame_no),1]/results_df['area_mm2']
    except: 
        print('No cross-section coordinates loaded - excluding width, area and stress from dataframe.')
    
    # drop rows with no cross-section listed
    results_df = results_df.dropna(axis=0, how = 'any')
    
    save_filename = 'results_df_frame_' + '{:02d}'.format(str(frame_no)) + '.pkl'
    results_df.to_pickle(os.path.join(dir_gom_results,save_filename))

#%% ===== INTERPOLATION DEBUGGING CODE =====
'''
good_triangle = []
triang = tri.Triangulation(df['x'], df['y'])
tr = triang.triangles
for row in range(tr.shape[0]):
    
    x1 = df['x'][tr[row,0]]
    x2 = df['x'][tr[row,1]]
    x3 = df['x'][tr[row,2]]
    y1 = df['y'][tr[row,0]]
    y2 = df['y'][tr[row,1]]
    y3 = df['y'][tr[row,2]]
    
    a = np.sqrt((x1-x2)**2 + (y1-y2)**2)
    b = np.sqrt((x1-x3)**2 + (y1-y3)**2)
    c = np.sqrt((x2-x3)**2 + (y2-y3)**2)
    
    sqa = a**2
    sqb = b**2
    sqc = c**2
    
    #if (sqa > sqc + sqb or sqb > sqa + sqc or sqc > sqa + sqb) and not ((a + b <= c) or (a + c <= b) or (b + c <= a)):
    # if (a + b <= c) or (a + c <= b) or (b + c <= a) : 
    if sqa < 0.2 and sqb < 0.2 and sqc < 0.2:
        good_triangle.append(0)
    else: 
        good_triangle.append(1)
        
max_radius = 7

triang = tri.Triangulation(df['x'], df['y'])
triangles = triang.triangles

fig1, ax1 = plt.subplots()
ax1.triplot(triang, 'o-',color = '#a4a4a4', mfc = '#383838', mec = '#383838', markersize=0.5, lw=0.5)

x = np.array(df['x'])
y = np.array(df['y'])

xtri = x[triangles] - np.roll(x[triangles], 1, axis=0)
ytri = y[triangles] - np.roll(x[triangles], 1, axis=0)
maxi = np.max(np.sqrt(xtri**2 + ytri**2), axis=1)

triang.set_mask(np.array(good_triangle) > 0)

ax1.triplot(triang, color='#3757b3', lw=0.5)
ax1.set_xlabel('x (mm)')
ax1.set_ylabel('y (mm)')
'''
