# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 11:53:48 2021

@author: jcv
"""
import pandas as pd
import numpy as np
from numpy.linalg import inv
from numpy.linalg import lstsq

def compute_KWW_parameters(df, var_y, var_x, end_idx, fit_range):
    '''Perform KWW fit to data and return relaxation constants
    
    Args: 
        df (dataframe): dataframe containing measured variables and time
        var_y (str): y variable to fit
        var_x (str): x variable to fit
        end_range (int): end index to use in computing average 'end' value
        fit_range (list/array): min and max inidces over which to perform
            linear fitting
            
    Returns:
        beta1: measure of narrowness of stretched exponential
        tau: characteristic relaxation time
    '''
    
    df['var_norm'] = (df[var_y] - df[var_y][end_idx:-1].mean())/(df[var_y].iloc[0] - df[var_y][end_idx:-1].mean())
    df['var_norm'] = df['var_norm'].apply(lambda x: np.abs(x))
    df['lnlny'] = df['var_norm'].apply(lambda x: np.log(np.log(1/x)))
    df['lnx'] = df[var_x].apply(lambda x: np.log(x))
    
    df.dropna(axis = 0, inplace = True)
    
    x = np.reshape(np.array(df['lnx'].iloc[fit_range[0]:fit_range[1]]), (-1,1))
    y = np.reshape(np.array(df['lnlny'].iloc[fit_range[0]:fit_range[1]]), (-1,1))

    # fit least squares regression model to data using normal equation approach
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
    
    return df, beta0, beta1, tau