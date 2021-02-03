#imports

import matplotlib as mp
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy import linalg as LA
import scipy as sp
#import pandas as pd
import sympy as sym
import os

#suporting math

#rotation matrices
def rx(theta):
    return np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])
def ry(theta):
    return np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
def rz(theta):
    return np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])

#code for rotating matrices
def rMat(mat, theta, axis = 'x'):
    if(axis == 'x' or axis == 1):
        return rx(theta) @ mat @ rx(-theta)
    if(axis == 'y' or axis == 2):
        return ry(theta) @ mat @ rz(-theta)
    if(axis == 'z' or axis == 3):
        return rz(theta) @ mat @ rz(-theta)

#coordinate transformations, no need for toCyl bc I don't do that, yet...
def toCar(vector):#vector has entries r, theta, z
    return np.array([vector[1]*np.cos(vector[0]),vector[1]*np.sin(vector[0]),vector[2]])

#levi-cevita symbol
LC = np.array([[[0,0,0],[0,0,1],[0,-1,0]],[[0,0,-1],[0,0,0],[1,0,0]],[[0,1,0],[-1,0,0],[0,0,0]]])

#code for fixing the aspect ratio on 3d plots: https://stackoverflow.com/questions/8130823/set-matplotlib-3d-plot-aspect-ratio
def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

# correction term for calculating radii of bumy particles, average z-coordinate over a unit hemisphere 
Rcorr = 3/4