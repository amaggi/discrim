import os,h5py
import numpy as np
import matplotlib.pyplot as plt

def generate_eq_catalog(n_eq, xmin, xmax, ymin, ymax):
    """
    Generate earthquakes uniformally distributed in time and space
    """

    x=np.random.uniform(xmin, xmax, size=n_eq)
    y=np.random.uniform(ymin, ymax, size=n_eq)
    t=np.random.uniform(0, 365, size=n_eq)
    h=np.random.uniform(0, 24, size=n_eq)
    d=np.ones(n_eq)
    d[h>18]=0
    d[h<6]=0
    # week and weekend
    # w=1 = week ; w=0 = weekend
    w_input=np.random.randint(0, 7, size=n_eq)
    w=np.ones(n_eq)
    w[w_input>4]=0

    eq=np.vstack((x,y,t,h,d,w))
    return eq.T

def generate_blast_catalog(n_blast,x_mine,y_mine,dim1,dim2,angle_deg,h,sigma_h):

    x=np.random.normal(loc=x_mine, scale=dim1, size=n_blast)
    y=np.random.normal(loc=y_mine, scale=dim2, size=n_blast)
    r=np.vstack((x,y))
    angle=np.radians(angle_deg)
    rotMatrix=np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
    r_rot=rotMatrix.dot(r)
    x_rot,y_rot=np.vsplit(r_rot,2)

    t=np.random.uniform(0, 365, size=n_blast)
    h=np.random.normal(loc=h, scale=sigma_h, size=n_blast)
    d=np.ones(n_blast)
    d[h>18]=0
    d[h<6]=0
    w_input=np.random.uniform(0,1,n_blast)
    w=np.ones(n_blast)
    w[w_input>0.999]=0

    blast=np.vstack((x_rot.flatten(), y_rot.flatten(), t, h, d, w))
    return blast.T

