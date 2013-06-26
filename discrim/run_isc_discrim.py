import os
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_fscore_support
from sklearn.naive_bayes import GaussianNB
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
from syn_catalog import *
from catalog_io_plot import *

# read dataset

#import pdb ; pdb.set_trace()

#bl_cat = read_isc_catalog('../CATALOGS/Loc_1987_2009_tri.txt')
#eq_cat = read_isc_catalog('../CATALOGS/Loc_1987_2009_tri_no.txt')
#write_catalogs_hdf5(eq_cat, bl_cat, '../CATALOGS/Loc_1987_2009_features.hdf5')

#X_tmp, y_tmp, features, labels = read_catalog_hdf5('../CATALOGS/Loc_1987_2009_features.hdf5')
X_tmp, y_tmp, features, labels = read_catalog_hdf5('ldg.hdf5')
# extract only Marlebach events
#x_min=6.5 ; x_max = 7.2 ; y_min = 49 ; y_max = 49.5

# extract larger region around Marlebach
x_min=5 ; x_max = 8 ; y_min = 48 ; y_max = 51
X=X_tmp[np.logical_and(np.logical_and(X_tmp[:,0]>x_min, X_tmp[:,0]<x_max), np.logical_and(X_tmp[:,1]>y_min, X_tmp[:,1]<y_max))]
y=y_tmp[np.logical_and(np.logical_and(X_tmp[:,0]>x_min, X_tmp[:,0]<x_max), np.logical_and(X_tmp[:,1]>y_min, X_tmp[:,1]<y_max))]

# use all events
#X=X_tmp
#y=y_tmp

min_cat = np.min(X, axis = 0)
max_cat = np.max(X, axis = 0)
plot_catalog(X, y, features, labels, 'Merlebach catalog', 'merlebach.png', ranges = (min_cat, max_cat))

basename='marlebach'
base_title='Marlebach'

scaler = preprocessing.StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

# split out a training set
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_scaled, y, test_size=0.5, random_state=0)
X_cv, X_test, y_cv, y_test = cross_validation.train_test_split(X_test, y_test, test_size=0.5, random_state=0)
n_cv=len(y_cv)
n_train=len(y_train)
min_cv=np.min(scaler.inverse_transform(X_cv), axis=0)
max_cv=np.max(scaler.inverse_transform(X_cv), axis=0)

plot_fname=basename + '_train.png'
title=base_title + ' training set' + '  ( %d samples )'%n_train
plot_catalog(scaler.inverse_transform(X_train), y_train, features, labels, title, plot_fname, ranges=(min_cv, max_cv))

plot_fname=basename + '_cv.png'
title=base_title + ' cross validation set' + '  ( %d samples )'%n_cv
plot_catalog(scaler.inverse_transform(X_cv), y_cv, features, labels, title, plot_fname, ranges=(min_cv, max_cv))

# do grid search
#print "doing grid search"
#C_range = 10.0 ** np.arange(-2, 5)
#gamma_range = 10.0 ** np.arange(-3,3)
#param_grid = dict(gamma=gamma_range, C=C_range)
#grid = GridSearchCV(svm.SVC(), param_grid=param_grid, n_jobs=-1)
#grid.fit(X_train, y_train)
#print("The best classifier is: ", grid.best_estimator_)


# do SVM
C=1.0
gamma=10.0
svc = svm.SVC(kernel='rbf',C=C, gamma=gamma, probability=True)
svc.fit(X_train, y_train)
y_pred=svc.predict(X_cv)
X_prob=svc.predict_proba(X_cv)

p,r,f,s = precision_recall_fscore_support(y_cv, y_pred)
print "Precision for SVM : ", p
print "Recall for SVM    : ", r
print "F1 for SVM        : ", f
n_false=np.sum(y_cv != y_pred)
frac_false=n_false/float(n_cv)
print "Number of mis-labeled points : %d"% n_false

X_false = X_cv[y_cv != y_pred]
y_false = y_cv[y_cv != y_pred]

X_prob_false=svc.predict_proba(X_false)

plot_fname=basename + '_svm_rbf.png'
title = base_title+' mis-labeled SVM RBF'+' ( %d / %d   %.2f %% )'%(n_false,n_cv,frac_false*100)
plot_catalog(scaler.inverse_transform(X_false), y_false, features, labels, title, plot_fname, ranges=(min_cv, max_cv))

plot_fname=basename + '_svm_rbf_prob.png'
title = base_title+' SVM RBF'+' probability'
plot_prob(scaler.inverse_transform(X_cv), X_prob, features, labels, title, plot_fname, ranges=(min_cv, max_cv))

plot_fname=basename + '_svm_rbf_false_prob.png'
title = base_title+' mis-labeled SVM RBF'+' probability  ( %d / %d   %.2f %% )'%(n_false,n_cv,frac_false*100)
plot_prob(scaler.inverse_transform(X_false), X_prob_false, features, labels, title, plot_fname, ranges=(min_cv, max_cv))



