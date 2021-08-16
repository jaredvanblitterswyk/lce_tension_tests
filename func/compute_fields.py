# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 16:22:47 2021

@author: jcv
"""
import os
import csv
import pandas as pd
import numpy as np
from numpy import linalg as LA
from scipy import linalg
import math as m
from scipy.interpolate import interp2d, griddata
import matplotlib.tri as tri

def compute_R(F):
    '''Compute rotation matrix from deformation gradient matrix
    
    Args: 
        F (array): deformation gradient
            
    Returns:
        An (array): rotation matrix
    '''
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
    '''Compute strain tensor and rotation matrix from displacements
    
    Args: 
        disp (dict): dictionary of displacement components
        strain_labels (list): strain components to compute
        spacing (array): pair of dx, dy spacings between interpolated points
            
    Returns:
        Eij (matrix): strain tensor
        Rij (matrix): rotation matrix
    '''
    
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
    #Reig = np.zeros((row,col))
    #stretch_p1 = np.zeros((row,col))
      
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
            
    return Eij, Rij

def mask_interp_region(triang, df, mask_side_len = 0.2):
    '''Mask specimen to remove poorly defined triangles in triangulation interp
    
    Args: 
        triang (object): object describing triangulation to X and Y coords
        df (dataframe): dataframe containing measurement values and coords
        mask_side_length (float, optional): maximum triangle side length 
            in masked area
            
    Returns:
        mask (object): object containing triangulation mask
        area_mask (array): contains area of masked elements for filtering
    '''
    
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
            
    return mask, area_mask

def interp_and_calc_strains(dir_results, file, mask_side_length, spacing, disp_labels, 
                            strain_labels, xx, yy, coord_trans_applied):
    '''Interpolate measurement points to pixel coords and compute disp and strain fields
    
    Args: 
        file (string): filename of gom results being processed
        mask_side_length (float): minimum side length of triangle for masking 
            ill-conditioned triangles
        spacing (array): pair of dx, dy spacings between interpolated points
        disp_labels (array): disp components to extract
        strain_labels (list): strain components to compute
        xx (matrix): grid of x pixel coordinates
        yy (matrix): grid of y pixel coordinates
        coord_trans_applied (bool): flag to determine if coordinate transform
            was applied in gom or if it is to be applied in function
            
    Returns:
        disp (dict): dictionary of arrays containing displacement fields
        Eij (matrix): strain tensor
        Rij (matrix): rotation matrix
        area_mask (array): contains area of masked elements for filtering
        triangle_mask (object): triangle mask object contianing coordiantes
            of triangles used to interpolate fields
    '''
    
    # load in data
    df = pd.read_csv(os.path.join(dir_results,file), skiprows = 5)
    
    df = df.dropna(axis = 0)
    df = df.reset_index(drop=True)
    # transform GOM coords back to reference configuration
    if coord_trans_applied:
        X = df['x.1'] - df['displacement_x']
        Y = df['y.1'] - df['displacement_y']
    else:
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
    Eij, Rij = calculateEijRot(disp, strain_labels, spacing)
    
    return disp, Eij, Rij, area_mask, triangle_mask

def calc_xsection(dir_xsection, filename, img_scale, orientation): 
    '''Compute cross-section of sample based on imageJ polygon coordinates
    
    Args: 
        dir_xsection (string): directory where polygon coordinate file located
        filename (string): name of coordinate file (csv)
        img_scale (float): mm per pixel scale for images
        orientation (string): denote 'horizontal' or 'vertical' orientation 
            of the sample with respect to the camera coordinates
            for example, cameras in standard mts test with cameras horizontal,
            specimen would be vertical - cameras rotated, specimen horizontal
            
    Returns:
        width_mm (array): width of sample at each pixel cross-section
    '''
    
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