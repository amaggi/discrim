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

def write_syn_catalogs_hdf5(eq_cat, blast_cat, filename):

    X = np.vstack((eq_cat,blast_cat))
    n_eq, f_f = eq_cat.shape
    n_s, n_f =  X.shape
    y = np.ones(n_s)
    y[0:n_eq]=0

    f=h5py.File(filename,'w')
    X_data=f.create_dataset('X',data=X)
    y_data=f.create_dataset('y',data=y)
    X_data.attrs['features'] = "x (km),y (km),Time (days),Hour,day/night,week"
    y_data.attrs['earthquake'] = 0
    y_data.attrs['blast'] = 1
    f.close()

def read_catalogs_hdf5(filename):

    f=h5py.File(filename,'r')
    X=np.array(f['X'])
    y=np.array(f['y'])
    features = f['X'].attrs['features'].split(',')
    labels={}
    for key, value in f['y'].attrs.iteritems():
        labels[key] = value
    f.close()

    return X, y, features, labels


def plot_catalogs(X, y, features, labels, title, filename):

    eq_label = labels['earthquake']
    bl_label = labels['blast']

    eq_cat =    X[y == eq_label]
    blast_cat = X[y == bl_label]

    fig = plt.figure()
    ax1=fig.add_subplot(221)
    ax2=fig.add_subplot(222)
    ax3=fig.add_subplot(223)
    ax4=fig.add_subplot(224)

    x_eq, y_eq, t_eq, h_eq, d_eq, w_eq = np.hsplit(eq_cat,6)
    x_blast, y_blast, t_blast, h_blast, d_blast, w_blast = np.hsplit(blast_cat,6)

    ax1.plot(eq_cat[w_eq.flatten()==1,2],eq_cat[w_eq.flatten()==1,3],'b.')
    ax1.plot(eq_cat[w_eq.flatten()==0,2],eq_cat[w_eq.flatten()==0,3],'.', color='green')
    ax1.plot(eq_cat[d_eq.flatten()==0,2],eq_cat[d_eq.flatten()==0,3],'k.')
    ax1.plot(blast_cat[w_blast.flatten()==1,2],blast_cat[w_blast.flatten()==1,3],'r.')
    ax1.plot(blast_cat[w_blast.flatten()==0,2],blast_cat[w_blast.flatten()==0,3],'.',color='yellow')
    ax1.plot(blast_cat[d_blast.flatten()==0,2],blast_cat[d_blast.flatten()==0,3],'.',color='cyan')
    #ax1.set_xlabel(features[2])
    ax1.set_ylabel(features[3])

    ax2.plot(eq_cat[w_eq.flatten()==1,0],eq_cat[w_eq.flatten()==1,3],'b.')
    ax2.plot(eq_cat[w_eq.flatten()==0,0],eq_cat[w_eq.flatten()==0,3],'g.')
    ax2.plot(eq_cat[d_eq.flatten()==0,0],eq_cat[d_eq.flatten()==0,3],'k.')
    ax2.plot(blast_cat[w_blast.flatten()==1,0],blast_cat[w_blast.flatten()==1,3],'r.')
    ax2.plot(blast_cat[w_blast.flatten()==0,0],blast_cat[w_blast.flatten()==0,3],'.',color='yellow')
    ax2.plot(blast_cat[d_blast.flatten()==0,0],blast_cat[d_blast.flatten()==0,3],'.',color='cyan')
    #ax2.set_xlabel(features[0])
    #ax2.set_ylabel(features[3])

    ax3.plot(eq_cat[w_eq.flatten()==1,2],eq_cat[w_eq.flatten()==1,1],'b.')
    ax3.plot(eq_cat[w_eq.flatten()==0,2],eq_cat[w_eq.flatten()==0,1],'g.')
    ax3.plot(eq_cat[d_eq.flatten()==0,2],eq_cat[d_eq.flatten()==0,1],'k.')
    ax3.plot(blast_cat[w_blast.flatten()==1,2],blast_cat[w_blast.flatten()==1,1],'r.')
    ax3.plot(blast_cat[w_blast.flatten()==0,2],blast_cat[w_blast.flatten()==0,1],'.',color='yellow')
    ax3.plot(blast_cat[d_blast.flatten()==0,2],blast_cat[d_blast.flatten()==0,1],'.',color='cyan')
    ax3.set_xlabel(features[2])
    ax3.set_ylabel(features[1])

    ax4.plot(eq_cat[w_eq.flatten()==1,0],eq_cat[w_eq.flatten()==1,1],'b.')
    ax4.plot(eq_cat[w_eq.flatten()==0,0],eq_cat[w_eq.flatten()==0,1],'g.')
    ax4.plot(eq_cat[d_eq.flatten()==0,0],eq_cat[d_eq.flatten()==0,1],'k.')
    ax4.plot(blast_cat[w_blast.flatten()==1,0],blast_cat[w_blast.flatten()==1,1],'r.')
    ax4.plot(blast_cat[w_blast.flatten()==0,0],blast_cat[w_blast.flatten()==0,1],'.',color='yellow')
    ax4.plot(blast_cat[d_blast.flatten()==0,0],blast_cat[d_blast.flatten()==0,1],'.',color='cyan')
    ax4.set_xlabel(features[0])
    #ax4.set_ylabel(features[1])

    plt.suptitle(title, fontsize=16)

    #plt.show()
    plt.savefig(filename)
    plt.clf()

