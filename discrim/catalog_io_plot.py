import os, h5py, string, re, glob
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil import tz

from_zone = tz.gettz('UTC')
to_zone =   tz.gettz('Europe/Paris')
base_date = datetime(1987,1,1)
base_date = base_date.replace(tzinfo=from_zone)
features = "x (km),y (km),Time (days),Hour,day/night,week"
labels = {'earthquake' : 0, 'blast' : 1}

def write_catalog_hdf5(X, y, filename):

    n_s, n_f =  X.shape

    f=h5py.File(filename,'w')
    X_data=f.create_dataset('X',data=X)
    y_data=f.create_dataset('y',data=y)
    X_data.attrs['features'] = features
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
    X_data.attrs['features'] = features
    y_data.attrs['earthquake'] = labels['earthquake']
    y_data.attrs['blast'] = labels['blast']
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

	# get time-related features
	t[i],h[i],d[i],w[i] = set_time_features_from_otime(otime)
        
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

	# get time-related features
	t[i],h[i],d[i],w[i] = set_time_features_from_otime(otime)

        i=i+1

    X = np.vstack((x,y,t,h,d,w))
    return X.T

def read_gse2_cat(filename):

    # read file
    f=open(filename,'r')
    lines=f.readlines()
    f.close()
    
    # count the number of events
    
    n_ev = 0
    for line in lines :
        if line.strip().startswith('EVENT'):
            n_ev += 1
            
    x = np.empty(n_ev, dtype=float) 
    y = np.empty(n_ev, dtype=float) 
    t = np.empty(n_ev, dtype=float) 
    h = np.empty(n_ev, dtype=float) 
    d = np.empty(n_ev, dtype=int) 
    w = np.empty(n_ev, dtype=int) 
    y_class = np.empty(n_ev, dtype=int) 
      
    i=0
    
    for line in lines :
        # skip empty lines
        if len( line.strip() ) == 0 :
            continue
        # if the first column is not empty
        if len( line[0].strip() ) > 0:
            # check if it is a date
            match_str = r'\d{4}/\d{2}/\d{2}\s+'
            if re.match(match_str, line.strip()):
		#print line
                # it is a date - this is the origin of the event
		words = line.split()

        	# parse date and time strings
        	# get origin time in UTC
        	otime=datetime.strptime(words[0]+' '+words[1], '%Y/%m/%d %H:%M:%S.%f')
        	otime = otime.replace(tzinfo=from_zone)

 		# get time-related features
		t1,h1,d1,w1 = set_time_features_from_otime(otime)

		# lat et lon
		# Note : do not use word decomposition as extraenous characters may be present
		lat_text = line[25:33]
		lon_text = line[34:43]
		# Some events may be detected but not located - must ignore them completely
		if len(lat_text.strip()) == 0 or len(lon_text.strip()) == 0 : 
		    ignore_next_type = True
		else : 
  		    lat = float(line[25:33])
		    lon = float(line[34:43])
		    ignore_next_type = False

        else :
            if not(line.strip().startswith('Date') or line.strip().startswith('rms') or line.strip().startswith('_ldg')):
                # TODO : Do parsing here
		words = line.split()
		ev_type = words[-1]	
		if ignore_next_type :
		    continue
                if ev_type == 'ke' or ev_type == 'km' :	
		    x[i] = lon
		    y[i] = lat
		    t[i] = t1
		    h[i] = h1
		    d[i] = d1
		    w[i] = w1
		    if ev_type == 'ke' :
                        y_class[i] = labels['earthquake']
		    else :
			y_class[i] = labels['blast']
		    i += 1
		else :
		   continue
    # There are now i elements in the vectors. So resize to length i.
    x.resize(i)
    y.resize(i)
    t.resize(i)
    h.resize(i)
    d.resize(i)
    w.resize(i)
    y_class.resize(i)

    X = np.vstack((x,y,t,h,d,w))
    return X.T, y_class
                
    # TODO : return catalog as for other inputs

            


def read_gse2_cat_from_directory(dir):
    X_list = []
    y_list = []
    files = glob.glob(dir + os.sep + '*.txt')
    for filename in files : 
	#print filename
	X, y = read_gse2_cat(filename)
	X_list.append(X)
	y_list.append(y)

    y_final = np.hstack((y_list))
    X_final = np.vstack((X_list))

    return X_final, y_final
    
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

def set_time_features_from_otime(otime):
    """
    Requires origin time in UTC. Returns t, h, d, w features.
    """

    # get origin time in local time
    otime_local = otime.astimezone(to_zone)
    #print otime.isoformat(' '), otime_local.isoformat(' ')

    # get t, fractional days since base_date
    t=((otime - base_date).days*86400 + (otime - base_date).seconds) / float(86400)

    # use local time to get h
    h=(otime_local.hour*3600+otime_local.minute*60+otime_local.second)/float(3600)


    # set day/night flag
    # for now cheat, and use 6-18h as daytime
    # TODO : use ephemerides to set day or night flag
    if h > 6.0 and h < 18.0 :
        d = 1
    else :
        d = 0

    # set weekday flag
    wday = otime_local.weekday()
    if wday < 5 : 
        w = 1
    else :
        w = 0
    
    return t, h, d, w
	

if __name__ == '__main__' :

    fname = '../CATALOGS/fiches_new_act.csv'
    mines = read_mine_cat(fname)
    plot_mines(mines, 'Mine short-list after J Frechet', 'Frechet_mines.png')
