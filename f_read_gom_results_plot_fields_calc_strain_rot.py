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
from numpy import linalg
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.font_manager as font_manager
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import interp2d, griddata
from scipy.linalg import sqrtm
import matplotlib.tri as tri
import numpy.ma as ma

# -- TO DO: would need to reformat m_process_dic_fields_btb_imaging
# currently not setup up properly for importing functions as a package

#sys.path.insert(0, 'Z:/Python/image_processing_shear_tests/')
#from m_process_dic_fields_btb_imaging import calculateEij
    
def calculateEijRot(disp_x, disp_y, dx, dy):

    I = np.matrix(([1,0],[0,1]))
    du_dy, du_dx = np.gradient(disp_x)
    dv_dy, dv_dx = np.gradient(disp_y)
    
    # gradient uses a unitless spacing of one, convert to mm
    du_dy = -1*du_dy/dy; dv_dy = -1*dv_dy/dy
    du_dx = du_dx/dx; dv_dx = dv_dx/dx
    
    row, col = disp_x.shape
    
    Eij = {'11': np.zeros((row,col)),'22': np.zeros((row,col)), '12': np.zeros((row,col))}
    Rij = np.zeros((row,col))
      
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
            
            eij = 0.5*(np.matmul(np.transpose(Fij),Fij)-I)
            #C = np.matmul(np.transpose(Fij),Fij)

            # calculate Lagrange strains
            Eij['11'][i,j] = eij[0,0]; Eij['22'][i,j] = eij[1,1]; Eij['12'][i,j] = 2*eij[0,1]
            
            # calculate eigenvalues and normalized eigenvectors
            #mask_nan = C == np.nan
            #C[mask_nan] = 0
            
            #U = sqrtm(C)
            '''
            eig = np.linalg.eig(C)
            e_val = eig[0]
            e_vec = eig[1]
            
            U = np.zeros((2,2))
            
            # calculate right stretch tensor
            U[0,0] = np.sum(e_val[0]*(e_vec[0,0]*e_vec[0,0]) + e_val[1]*e_vec[0,1]*e_vec[0,1])
            U[0,1] = np.sum(e_val[0]*(e_vec[0,0]*e_vec[1,0]) + e_val[1]*e_vec[1,0]*e_vec[1,1])
            U[1,0] = np.sum(e_val[0]*(e_vec[0,0]*e_vec[1,0]) + e_val[1]*e_vec[1,0]*e_vec[1,1])
            U[1,1] = np.sum(e_val[0]*(e_vec[0,1]*e_vec[0,1]) + e_val[1]*e_vec[1,1]*e_vec[1,1])
            '''
            #eij = 0.5*(np.matmul(U,np.transpose(U))-I)
            
            #Eij['11'][i,j] = eij[0,0]; Eij['22'][i,j] = eij[1,1]; Eij['12'][i,j] = 2*eij[0,1]
            '''
            # calculate inverse sqr. root
            Cnij = np.zeros((2,2))
            
            Cnij[0,0] = np.sum(1/e_val[0]*(e_vec[0,0]*e_vece_vec[0,0]) + 1/e_val[1]*e_vec[0,1]*e_vec[0,1])
            Cnij[0,1] = np.sum(1/e_val[0]*(e_vec[0,0]*e_vece_vec[1,0]) + 1/e_val[1]*e_vec[1,0]*e_vec[1,1])
            Cnij[1,0] = np.sum(1/e_val[0]*(e_vec[0,0]*e_vece_vec[1,0]) + 1/e_val[1]*e_vec[1,0]*e_vec[1,1])
            Cnij[1,1] = np.sum(1/e_val[0]*(e_vec[0,1]*e_vece_vec[0,1]) + 1/e_val[1]*e_vec[1,1]*e_vec[1,1])
            
            # compute rotation matrix
            Rij[i,j] = np.matmul(Fij,Cnij)[0,0]
            '''
    return Eij#, Rij

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
            
    return triangle_mask

def plot_field_contour_save(xx,yy,zz,vmin,vmax,cmap,level_boundaries,fpath,hide_labels):
    # plot map
    f = plt.figure(figsize = (8,2))
    ax = f.add_subplot(1,1,1)
    cf = ax.contourf(xx,yy,zz,level_boundaries,vmin = vmin,vmax = vmax, cmap = cmap)
    #ax.scatter(df['x'],df['y'],s = 1, c = df['epsilon_x']/100, cmap = custom_map, alpha = 0.2)
    ax.set_xlim([xmin,xmax])
    ax.set_ylim([ymin,ymax])
    ax.grid(True)
    cbar = f.colorbar(cf)

    # show grid but hide labels
    if hide_labels:
        ax.xaxis.set_ticklabels([])
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticklabels([])
        ax.yaxis.set_ticks_position('none')    
    else:
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        
    plt.tight_layout()  
    # plt.show()
    plt.ioff() # turn off interactive mode
    
    # save figure
    f.savefig(fpath, dpi=300, facecolor='w', edgecolor='w', pad_inches=0.1)
    
def load_and_plot(file, mask_side_length, dx, dy, hide_labels, calc_strain_rot):
    # load in data
    df = pd.read_csv(os.path.join(dir_results,file), skiprows = 5)
    
    # transform GOM coords back to reference configuration
    X = df['x']+(Nx/2)*img_scale - df['displacement_x']
    Y = df['y']+(Ny/2)*img_scale - df['displacement_y']
      
    # extract frame number for saving image
    frame_no = file[35:-10]    
    print('Processing frame:' + str(frame_no))
    # define triangulation interpolation on x and y coordinates frm GOM
    # in reference coordinates
    triang = tri.Triangulation(X, Y)
    # get mask
    triangle_mask = mask_interp_region(triang, df, mask_side_length)
    # apply mask
    triang.set_mask(np.array(triangle_mask) > 0)
       
    # determine if strains to be calculated manually or not
    if calc_strain_rot:
        
        # TO DO: add plotting functionality for displacements
        # alternate approach: have one script to read in and process ux, uy
        # and another to handle strains (calculated or extracted from gom)
        for var in ['displacement_x','displacement_y']:
            dir_save_figs = os.path.join(dir_figs_root,'disp_fields')
            interpolator = tri.LinearTriInterpolator(triang, df[var])
            
            # evaluate interpolator object at regular grids
            if var == 'displacement_x':
                vmin, vmax = 0, 21
                var_fname = 'ux'
                ux = interpolator(xx, yy)
            else:
                vmin, vmax = -1.2, 0
                var_fname = 'uy'
                uy = interpolator(xx, yy)
                    
        Eij = calculateEijRot(ux,uy, img_scale, img_scale)
        
        # loop through and plot each strain component
        for strain_component in ['11','22','12']:
            if strain_component == '11':
                vmin, vmax = 0, 4.5
            elif strain_component == '22':
                vmin, vmax = -0.4, 0
            else:
                vmin, vmax = -0.5, 1
            
            # define contour map level boundaries
            level_boundaries = np.linspace(vmin,vmax,levels+1)
            
            #define variable to plot - interpolated to deformed coordinates
            zz = Eij[strain_component]
            
            # specify variable name
            var_fname = 'E'+strain_component+'_calc'
            # define directory to save to
            dir_save_figs = os.path.join(dir_figs_root,'strain_fields')
            # define full image path
            fpath = dir_save_figs+'/'+spec_id+'_'+var_fname+'_'+frame_no+'.tiff'
            # generate figure
            plot_field_contour_save(xx+ux,yy+uy,zz,vmin,vmax,custom_map,level_boundaries,fpath,hide_labels)
        
    else:
        for var in ['displacement_x','displacement_y','epsilon_x','epsilon_y','epsilon_xy']:  
            # define masked interpolator object
            if var in ['epsilon_x','epsilon_y']:
                dir_save_figs = os.path.join(dir_figs_root,'strain_fields')
                interpolator = tri.LinearTriInterpolator(triang, df[var]/100)
                if var == 'epsilon_x':
                    vmin, vmax = 0, 4.5
                    var_fname = 'Exx'
                else:
                    vmin, vmax = -0.4, 0
                    var_fname = 'Eyy'
            elif var in ['epsilon_xy']:
                dir_save_figs = os.path.join(dir_figs_root,'strain_fields')
                interpolator = tri.LinearTriInterpolator(triang, df[var]*2)
                vmin, vmax = -0.5, 1
                var_fname = 'Exy'
            else:
                dir_save_figs = os.path.join(dir_figs_root,'disp_fields')
                interpolator = tri.LinearTriInterpolator(triang, df[var])
                if var == 'displacement_x':
                    vmin, vmax = 0, 21
                    var_fname = 'ux'
                else:
                    vmin, vmax = -1.2, 0
                    var_fname = 'uy'
            
            # define contour map level boundaries
            level_boundaries = np.linspace(vmin,vmax,levels+1)
            # evaluate interpolator object at regular grids
            zz = interpolator(xx, yy)
            # define full image path
            fpath = dir_save_figs+'/'+spec_id+'_'+var_fname+'_'+frame_no+'.tiff'
            
            # generate figure
            plot_field_contour_save(xx,yy,zz,vmin,vmax,custom_map,fpath,hide_labels)
    
#%% ---- MAIN SCRIPT ----
dir_results = 'Z:/Experiments/lce_tension/lcei_001/007_t01_r01/gom_results'

files = [f for f in os.listdir(dir_results) if f.endswith('.csv')]
spec_id = 'lcei_001_007_t01_r00'
avg_shear_strain = []
'''
for i in range(0,len(files)):
    df = pd.read_csv(os.path.join(dir_results,files[i]), skiprows = 5)
    avg_shear_strain.append(df['epsilon_x'].mean())
'''
img_scale = 0.0106 # mm/pix
    
# range of corrdinates to consider for plotting
xmin,xmax = 0, 26
ymin, ymax = 8, 12

# number of pixels in x and y dimensions 
Nx = 2448
Ny = 2048

img_scale = 0.0106

# create vector of x dimensions
x_vec = np.linspace(0,Nx-1,Nx)*img_scale # vector of original x coordinates (pixels)
y_vec = np.linspace(Ny-1,0,Ny)*img_scale # vector of original y coordinates (pixels)

xx,yy = np.meshgrid(x_vec,y_vec)

# calculate scaled spacing between points in field
dx = img_scale
dy = img_scale

# colorbar levels
levels = 25

# load in colormap
cm_data = np.loadtxt("Z:/Python/mpl_styles/lajolla.txt")
custom_map = LinearSegmentedColormap.from_list('custom', cm_data)

# max side length of triangles in DeLauny triangulation 
mask_side_length = 0.5

# ----- configure directories -----
# check if figures folder exists, if not, make directory
if not os.path.exists(os.path.join(dir_results,'figures')):
    os.makedirs(os.path.join(dir_results,'figures'))

# define directory where figures to be saved    
dir_figs_root = os.path.join(dir_results,'figures')

if not os.path.exists(os.path.join(dir_figs_root,'disp_fields')):
    os.makedirs(os.path.join(dir_figs_root,'disp_fields'))
    
if not os.path.exists(os.path.join(dir_figs_root,'strain_fields')):
    os.makedirs(os.path.join(dir_figs_root,'strain_fields'))

# ---- run processing -----    
for i in range(0,len(files)):
    load_and_plot(files[i], mask_side_length, dx, dy, hide_labels = False, calc_strain_rot = True)

#%% ===== DEBUGGING CODE =====
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
