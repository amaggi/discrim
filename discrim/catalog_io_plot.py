import os, h5py, string
import numpy as np
import matplotlib.pyplot as plt

def write_catalogs_hdf5(X, y, features, labels):

    X = np.vstack((eq_cat,blast_cat))
    n_eq, f_f = eq_cat.shape
    n_s, n_f =  X.shape
    y = np.ones(n_s)
    y[0:n_eq]=0

    f=h5py.File(filename,'w')
    X_data=f.create_dataset('X',data=X)
    y_data=f.create_dataset('y',data=y)
    X_data.attrs['features'] = string.join(features,',')
    y_data.attrs['earthquake'] = labels['earthquake']
    y_data.attrs['blast'] = labels['blast']
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

