# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 14:51:36 2021

@author: jcv
"""
import numpy.linalg as LA
import math as m

# define manual test case
F = np.array([[1, 0.495, 0.5],
              [-0.333, 1, -0.247],
              [0.959, 0, 1.5]
              ])

def compute_R(F):
    # process based on iterative technique presented in:
    # http://www.continuummechanics.org/polardecomposition.html
    
    # compute transpose of inverse
    Fti = LA.inv(F.T)
    
    # compute norm of matrix - iterative way of computing R instead of matrix 
    # sqrt calc
    Ao = np.zeros((F.shape[0],F.shape[1]))
    An = 0.5*(F + Fti)
    # define the difference between matrices using the normalized distance
    res = LA.norm(An - Ao)
    
    threshold = 0.1
    while res > threshold:
        Ao = An
        Ati = LA.inv(Ao.T)
        An = 0.5*(Ao + Ati)
        res = LA.norm(An - Ao)
        
    return An

R = compute_R(F)

angle = np.cos(R[0,0])*180/m.pi
