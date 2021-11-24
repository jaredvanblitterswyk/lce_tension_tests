# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 11:53:48 2021

@author: jcv
"""
import pandas as pd
import numpy as np
from numpy.linalg import inv
from numpy.linalg import lstsq

def compute_KWW_parameters(df, fit_range):
    '''Perform KWW fit to data and return relaxation constants
    
    Args: 
        df (dataframe): dataframe containing measured variables and time
        fit_range (list/array): min and max inidces over which to perform
            linear fitting
            
    Returns:
        beta0: intercept
        beta1: slope - measure of narrowness of stretched exponential
        tau: characteristic relaxation time
    '''
   
    x = np.reshape(np.array(df['lnx'].iloc[fit_range[0]:fit_range[1]]), (-1,1))
    y = np.reshape(np.array(df['lnlny'].iloc[fit_range[0]:fit_range[1]]), (-1,1))

    # fit least squares regression model to data using normal equation approach
    try:
        xo = np.ones(x.shape)
        X = np.matrix(np.stack((xo, x), axis = -1))
        Xt = np.transpose(X)
        Xty = Xt*y
        
        # compute fitted model parameters
        beta = inv(Xt*X)*Xty     
        
        # extract intercept and slope
        beta0 = beta[0].item()
        beta1 = beta[1].item()
        tau = np.exp(-beta0/beta1)
    except:
        beta0 = 0
        beta1 = 0
        tau = 0
    
    return beta0, beta1, tau