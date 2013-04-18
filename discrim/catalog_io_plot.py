import os, h5py, string
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil import tz

from_zone = tz.gettz('UTC')
to_zone =   tz.gettz('Europe/Paris')
base_date = datetime(1987,1,1)
base_date = base_date.replace(tzinfo=from_zone)

def write_catalog_hdf5(X, y, features, labels):

    X = np.vstack((eq_cat,blast_cat))
    n_eq, f_f = eq_cat.shape
    n_s, n_f =  X.shape
    # set labels : 0 = eq, 1=mine blast
    y = np.ones(n_s)
    y[0:n_eq]=0

    f=h5py.File(filename,'w')
    X_data=f.create_dataset('X',data=X)
    y_data=f.create_dataset('y',data=y)
    X_data.attrs['features'] = string.join(features,',')
    y_data.attrs['earthquake'] = labels['earthquake']
    y_data.attrs['blast'] = labels['blast']
    f.close()

def read_catalog_hdf5(filename):

    f=h5py.File(filename,'r')
    X=np.array(f['X'])
    y=np.array(f['y'])
    features = f['X'].attrs['features'].split(',')
    labels={}
    for key, value in f['y'].attrs.iteritems():
        labels[key] = value
    f.close()

    return X, y, features, labels

def write_catalogs_hdf5(eq_cat, blast_cat, filename):

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


def read_isc_catalog(filename):

    # read file
    f=open(filename,'r')
    lines=f.readlines()
    f.close()

    # set up storage
    ns = len(lines)
    x = np.empty(len(lines), dtype=float) 
    y = np.empty(len(lines), dtype=float) 
    t = np.empty(len(lines), dtype=float) 
    h = np.empty(len(lines), dtype=float) 
    d = np.empty(len(lines), dtype=int) 
    w = np.empty(len(lines), dtype=int) 

    i=0
    for line in lines:
        words = line.split()
        #print words

        # get longitude and latitude (interpret as x and y for now)
        # TODO : project onto a local coordinate system
        lon = float(words[2])
        lat = float(words[3])
        x[i]=lon
        y[i]=lat

        # parse date and time strings
        year = int(words[0][0:2])
        if year > 70 :
            year = year + 1900
        else :
            year = year + 2000
        month = int(words[0][2:4])
        day   = int(words[0][4:6])
        hour  = int(words[1][0:2])
        mins  = int(words[1][2:4])
        sec   = int(words[1][4:6])
        try:
            usec  = int(words[1][7:9])*10000
        except ValueError :
            usec = 0

        # get origin time in UTC
        otime=datetime(year, month, day, hour, mins, sec, usec)
        otime = otime.replace(tzinfo=from_zone)

        # get origin time in local time
        otime_local = otime.astimezone(to_zone)
        #print otime.isoformat(' '), otime_local.isoformat(' ')

        # get t, fractional days since base_date
        t[i]=((otime - base_date).days*86400 + (otime - base_date).seconds) / float(86400)

        # use local time to get h
        h[i]=(otime_local.hour*3600+otime_local.minute*60+otime_local.second)/float(3600)


        # set day/night flag
        # for now cheat, and use 6-18h as daytime
        # TODO : use ephemerides to set day or night flag
        if h[i] > 6.0 and h[i] < 18.0 :
            d[i] = 1
        else :
            d[i] = 0

        # set weekday flag
        wday = otime_local.weekday()
        if wday < 5 : 
            w[i] = 1
        else :
            w[i] = 0
        
        i=i+1

    X = np.vstack((x,y,t,h,d,w))
    return X.T

def read_chooz_catalog(filename):

    # read file
    f=open(filename,'r')
    lines=f.readlines()
    f.close()

    # set up storage
    ns = len(lines)
    x = np.empty(len(lines), dtype=float) 
    y = np.empty(len(lines), dtype=float) 
    t = np.empty(len(lines), dtype=float) 
    h = np.empty(len(lines), dtype=float) 
    d = np.empty(len(lines), dtype=int) 
    w = np.empty(len(lines), dtype=int) 

    i=0
    for line in lines:
        words = line.split(',')
        #print words

        # get longitude and latitude (interpret as x and y for now)
        # TODO : project onto a local coordinate system
        lat = float(words[2])
        lon = float(words[3])
        x[i]=lon
        y[i]=lat

        # parse date and time strings
        # get origin time in UTC
        otime=datetime.strptime(words[0]+' '+words[1], '%Y-%m-%d %H:%M:%S.%f')
        otime = otime.replace(tzinfo=from_zone)

        # get origin time in local time
        otime_local = otime.astimezone(to_zone)
        #print otime.isoformat(' '), otime_local.isoformat(' ')

        # get t, fractional days since base_date
        t[i]=((otime - base_date).days*86400 + (otime - base_date).seconds) / float(86400)

        # use local time to get h
        h[i]=(otime_local.hour*3600+otime_local.minute*60+otime_local.second)/float(3600)


        # set day/night flag
        # for now cheat, and use 6-18h as daytime
        # TODO : use ephemerides to set day or night flag
        if h[i] > 6.0 and h[i] < 18.0 :
            d[i] = 1
        else :
            d[i] = 0

        # set weekday flag
        wday = otime_local.weekday()
        if wday < 5 : 
            w[i] = 1
        else :
            w[i] = 0
        
        i=i+1

    X = np.vstack((x,y,t,h,d,w))
    return X.T

def read_mine_cat(filename):

    f=open(filename,'r')
    lines=f.readlines()
    f.close()

    n_mines=len(lines[2:-1])

    mines={}
    mine_id=np.empty(n_mines, dtype=int)
    mine_xy=np.empty((n_mines,2),dtype=float)
    mine_cp=[]
    mine_name=[]

    i=0
    for line in lines[2:-1]:
        words = line.split('\t')
        mine_id[i] = int(words[0])
        mine_xy[i,0] = float(words[7])/1000.0
        mine_xy[i,1] = float(words[8])/1000.0
        mine_name.append(words[1])
        mine_cp.append(words[4])
        i=i+1
    mines['ID'] = mine_id
    mines['CP'] = mine_cp
    mines['name'] = mine_name
    mines['xy'] = mine_xy

    return mines

def plot_catalog(X, y, features, labels, title, filename, ranges=None):

    eq_label = labels['earthquake']
    bl_label = labels['blast']

    eq_cat =    X[y == eq_label]
    blast_cat = X[y == bl_label]

    fig = plt.figure(figsize=(9.5, 4))
    ax1=fig.add_subplot(121, axisbg='lightgrey')
    ax2=fig.add_subplot(122, axisbg='lightgrey')
    #ax3=fig.add_subplot(223, axisbg='lightgrey')
    #ax4=fig.add_subplot(224, axisbg='lightgrey')

    x_eq, y_eq, t_eq, h_eq, d_eq, w_eq = np.hsplit(eq_cat,6)
    x_blast, y_blast, t_blast, h_blast, d_blast, w_blast = np.hsplit(blast_cat,6)

    w_eq = np.around(w_eq.flatten())
    d_eq = np.around(d_eq.flatten())
    w_blast = np.around(w_blast.flatten())
    d_blast = np.around(d_blast.flatten())


    ax1.scatter(eq_cat[w_eq==1,2],eq_cat[w_eq==1,3],marker='o', color='b', linewidths=(0,))
    ax1.scatter(eq_cat[w_eq==0,2],eq_cat[w_eq==0,3],marker='o', color='green', linewidths=(0,))
    ax1.scatter(eq_cat[d_eq==0,2],eq_cat[d_eq==0,3],marker='o', color='black', linewidths=(0,))
    ax1.scatter(blast_cat[w_blast==1,2],blast_cat[w_blast==1,3], marker='o',color='red',  linewidths=(0,))
    ax1.scatter(blast_cat[w_blast==0,2],blast_cat[w_blast==0,3], marker='o',color='yellow', linewidths=(0,))
    ax1.scatter(blast_cat[d_blast==0,2],blast_cat[d_blast==0,3], marker='o',color='cyan', linewidths=(0,))
    ax1.set_xlabel(features[2])
    ax1.set_ylabel(features[3])
    if ranges:
        ax1.set_xlim(ranges[0][2], ranges[1][2])
        ax1.set_ylim(ranges[0][3], ranges[1][3])

    #ax2.plot(eq_cat[w_eq==1,0],eq_cat[w_eq==1,3],'b.')
    #ax2.plot(eq_cat[w_eq==0,0],eq_cat[w_eq==0,3],'g.')
    #ax2.plot(eq_cat[d_eq==0,0],eq_cat[d_eq==0,3],'k.')
    #ax2.plot(blast_cat[w_blast==1,0],blast_cat[w_blast==1,3],'r.')
    #ax2.plot(blast_cat[w_blast==0,0],blast_cat[w_blast==0,3],'.',color='yellow')
    #ax2.plot(blast_cat[d_blast==0,0],blast_cat[d_blast==0,3],'.',color='cyan')
    #ax2.set_xlabel(features[0])
    #ax2.set_ylabel(features[3])

    #ax3.plot(eq_cat[w_eq==1,2],eq_cat[w_eq==1,1],'b.')
    #ax3.plot(eq_cat[w_eq==0,2],eq_cat[w_eq==0,1],'g.')
    #ax3.plot(eq_cat[d_eq==0,2],eq_cat[d_eq==0,1],'k.')
    #ax3.plot(blast_cat[w_blast==1,2],blast_cat[w_blast==1,1],'r.')
    #ax3.plot(blast_cat[w_blast==0,2],blast_cat[w_blast==0,1],'.',color='yellow')
    #ax3.plot(blast_cat[d_blast==0,2],blast_cat[d_blast==0,1],'.',color='cyan')
    #ax3.set_xlabel(features[2])
    #ax3.set_ylabel(features[1])

    ax2.scatter(eq_cat[w_eq==1,0],eq_cat[w_eq==1,1], marker='o', color='b', linewidths=(0,))
    ax2.scatter(eq_cat[w_eq==0,0],eq_cat[w_eq==0,1],marker='o',color='green', linewidths=(0,))
    ax2.scatter(eq_cat[d_eq==0,0],eq_cat[d_eq==0,1],marker='o',color='black', linewidths=(0,))
    ax2.scatter(blast_cat[w_blast==1,0],blast_cat[w_blast==1,1],marker='o', color='red', linewidths=(0,))
    ax2.scatter(blast_cat[w_blast==0,0],blast_cat[w_blast==0,1],marker='o',color='yellow', linewidths=(0,))
    ax2.scatter(blast_cat[d_blast==0,0],blast_cat[d_blast==0,1],marker='o',color='cyan', linewidths=(0,))
    ax2.set_xlabel(features[0])
    ax2.set_ylabel(features[1])
    if ranges:
        ax2.set_xlim(ranges[0][0], ranges[1][0])
        ax2.set_ylim(ranges[0][1], ranges[1][1])

    fig.subplots_adjust(hspace=4)

    plt.suptitle(title, fontsize=16)

    #plt.show()
    plt.savefig(filename)
    plt.clf()


def plot_prob(X, X_prob, features, labels, title, filename, ranges=None):

    fig = plt.figure(figsize=(9.5, 4))
    ax1=fig.add_subplot(121, axisbg='lightgrey')
    ax2=fig.add_subplot(122, axisbg='lightgrey')
    #ax3=fig.add_subplot(223, axisbg='lightgrey')
    #ax4=fig.add_subplot(224, axisbg='lightgrey')

    x, y, t, h, d, w = np.hsplit(X,6)
    prob=np.max(X_prob, axis=1)

    w = np.around(w.flatten())
    d = np.around(d.flatten())


    ax1.scatter(t.flatten(),h.flatten(),c=prob, marker='o', linewidths=(0,), cmap='afmhot')
    ax1.set_xlabel(features[2])
    ax1.set_ylabel(features[3])
    if ranges:
        ax1.set_xlim(ranges[0][2], ranges[1][2])
        ax1.set_ylim(ranges[0][3], ranges[1][3])

    cs=ax2.scatter(x.flatten(),y.flatten(),c=prob, marker='o', linewidths=(0,), cmap='afmhot')
    ax2.set_xlabel(features[0])
    ax2.set_ylabel(features[1])
    if ranges:
        ax2.set_xlim(ranges[0][0], ranges[1][0])
        ax2.set_ylim(ranges[0][1], ranges[1][1])

    fig.colorbar(cs, ax=ax2, shrink=0.9)
    fig.subplots_adjust(hspace=4)

    plt.suptitle(title, fontsize=16)

    #plt.show()
    plt.savefig(filename)
    plt.clf()

def dot2(u, v):
    return u[0]*v[0] + u[1]*v[1]

def cross2(u, v, w):
    """u x (v x w)"""
    return dot2(u, w)*v - dot2(u, v)*w

def ncross2(u, v):
    """|| u x v ||^2"""
    return sq2(u)*sq2(v) - dot2(u, v)**2

def sq2(u):
    return dot2(u, u)


def plot_mines(mines_dict, title, filename):
    from scipy.spatial import Delaunay
    from matplotlib.collections import LineCollection

    fig = plt.figure(figsize=(9.5,4))
    ax1=fig.add_subplot(121, axisbg='lightgrey')
    ax2=fig.add_subplot(122, axisbg='lightgrey')
    
    # plot scatterplot of mines
    x=mines_dict['xy'][:,0]
    y=mines_dict['xy'][:,1]
    ax1.scatter(x,y,marker='.',color='red', linewidths=(0,))
    ax1.set_xlim(np.min(x), np.max(x))
    ax1.set_ylim(np.min(y), np.max(y))

    # plot voronoi cells
    tri=Delaunay(mines_dict['xy'])
    p = tri.points[tri.vertices]

    # Triangle vertices
    A = p[:,0,:].T
    B = p[:,1,:].T
    C = p[:,2,:].T

    # See http://en.wikipedia.org/wiki/Circumscribed_circle#Circumscribed_circles_of_triangles
    # The following is just a direct transcription of the formula there
    a = A - C
    b = B - C

    cc = cross2(sq2(a) * b - sq2(b) * a, a, b) / (2*ncross2(a, b)) + C

    # Grab the Voronoi edges
    vc = cc[:,tri.neighbors]
    vc[:,tri.neighbors == -1] = np.nan # edges at infinity, plotting those would need more work...

    lines = []
    lines.extend(zip(cc.T, vc[:,:,0].T))
    lines.extend(zip(cc.T, vc[:,:,1].T))
    lines.extend(zip(cc.T, vc[:,:,2].T))

    # Plot it

    lines = LineCollection(lines, edgecolor='k', linewidth=0.2)

    #ax2.plot(points[:,0], points[:,1], '.')
    #ax2.plot(cc[0], cc[1], '.')
    ax2.set_xlim(np.min(x), np.max(x))
    ax2.set_ylim(np.min(y), np.max(y))
    plt.gca().add_collection(lines)


    plt.suptitle(title, fontsize=16)
    plt.savefig(filename)
    plt.clf()

if __name__ == '__main__' :

    fname = '../CATALOGS/fiches_new_act.csv'
    mines = read_mine_cat(fname)
    plot_mines(mines, 'Mine short-list after J Frechet', 'Frechet_mines.png')
