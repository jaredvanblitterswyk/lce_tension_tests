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
            
            # calculate Lagrange strains
            Eij['11'][i,j] = eij[0,0]; Eij['22'][i,j] = eij[1,1]; Eij['12'][i,j] = 2*eij[0,1]
            
            # calculate rotation - need inverse sqr. root of C
            C = np.matmul(np.transpose(Fij),Fij)
            
            # calculate eigenvalues and normalized eigenvectors
            eig = np.linalg.eig(C)
            e_val = eig[0]
            e_vec = eig[1]
            
            # calculate inverse sqr. root
            Cnij = np.zeros((2,2))
            
            Cnij[0,0] = np.sum(1/e_val[0]*(e_vec[0,0]*e_vece_vec[0,0]) + 1/e_val[1]*e_vec[0,1]*e_vec[0,1])
            Cnij[0,1] = np.sum(1/e_val[0]*(e_vec[0,0]*e_vece_vec[1,0]) + 1/e_val[1]*e_vec[1,0]*e_vec[1,1])
            Cnij[1,0] = np.sum(1/e_val[0]*(e_vec[0,0]*e_vece_vec[1,0]) + 1/e_val[1]*e_vec[1,0]*e_vec[1,1])
            Cnij[1,1] = np.sum(1/e_val[0]*(e_vec[0,1]*e_vece_vec[0,1]) + 1/e_val[1]*e_vec[1,1]*e_vec[1,1])
            
            # compute rotation matrix
            Rij = np.matmul(Fij,Cnij)
            
    return Eij, Rij

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
    
def load_and_plot(file, mask_side_length, hide_labels):
    
    # load in data
    df = pd.read_csv(os.path.join(dir_results,file), skiprows = 5)
    
    # extract frame number for saving image
    frame_no = file[35:-10]    
    print('Processing frame:' + str(frame_no))
    # define triangulation interpolation on x and y coordinates frm GOM
    triang = tri.Triangulation(df['x'], df['y'])
    # get mask
    triangle_mask = mask_interp_region(triang, df, mask_side_length)
    # apply mask
    triang.set_mask(np.array(triangle_mask) > 0)
    
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
        
        level_boundaries = np.linspace(vmin,vmax,levels+1)
        # evaluate interpolator object at regular grids
        zz = interpolator(xx, yy)
    
        # plot map
        f = plt.figure(figsize = (8,2))
        ax = f.add_subplot(1,1,1)
        cf = ax.contourf(xx,yy,zz,level_boundaries,vmin = vmin,vmax = vmax, cmap = custom_map)
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
        plt.ioff() # turn off interactive mode
        #plt.show()
        
        # save figure
        fpath = dir_save_figs+'/'+spec_id+'_'+var_fname+'_'+frame_no+'.tiff'
        f.savefig(fpath, dpi=300, facecolor='w', edgecolor='w', pad_inches=0.1)
    
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
xmin,xmax = -15, 15
ymin, ymax = -3.5, 2
x_vec = np.linspace(xmin,xmax,900)
y_vec = np.linspace(ymin,ymax,200) 
xx,yy = np.meshgrid(x_vec,y_vec)

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
    load_and_plot(files[i], mask_side_length, hide_labels = False)

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
