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

#################
# create datasets
#################

# Trimouns
filename='trimouns.hdf5'
eq_cat=generate_eq_catalog(1000,-30,30,-30,30)
blast_cat=generate_blast_catalog(1000,0,0,3,3,30,17,0.1)
write_catalogs_hdf5(eq_cat, blast_cat, filename)
X, y, features, labels = read_catalog_hdf5(filename)
min_cat=np.min(X, axis=0)
max_cat=np.max(X, axis=0)
plot_catalog(X, y, features, labels, 'Trimouns synthetic catalog', 'trimouns_syn.png', ranges=(min_cat,max_cat))


# Mollard
filename='mollard.hdf5'
eq_cat=generate_eq_catalog(1000,-40,40,-40,40)
blast_cat=generate_blast_catalog(1000,0,0,10,10,30,12,2)
write_catalogs_hdf5(eq_cat, blast_cat, filename)
X, y, features, labels = read_catalog_hdf5(filename)
min_cat=np.min(X, axis=0)
max_cat=np.max(X, axis=0)
plot_catalog(X, y, features, labels, 'Mollard synthetic catalog', 'mollard_syn.png', ranges=(min_cat,max_cat))

# skew
filename='skew.hdf5'
eq_cat=generate_eq_catalog(1000,-40,40,-40,40)
blast_cat=generate_blast_catalog(1000,0,0,5,15,30,12,2)
write_catalogs_hdf5(eq_cat, blast_cat, filename)
X, y, features, labels = read_catalog_hdf5(filename)
min_cat=np.min(X, axis=0)
max_cat=np.max(X, axis=0)
plot_catalog(X, y, features, labels, 'Skew synthetic catalog', 'skew_syn.png', ranges=(min_cat,max_cat))

#exit()

###########################
# read and pre-process data
###########################

# eq = 0
# blast = 1

filenames=['trimouns.hdf5', 'mollard.hdf5', 'skew.hdf5']
base_titles=['Trimouns', 'Mollard', 'Skew']
C_values=[10000, 10, 10000]
gamma_values=[0.01, 0.1, 0.01]

for filename, base_title, C, gamma in zip(filenames, base_titles, C_values, gamma_values) : 
    print filename, base_title

    basename,ext=os.path.splitext(filename)
    X, y, features, labels = read_catalog_hdf5(filename)

    # do feature scaling
    scaler = preprocessing.StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)

    # split out a training set
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_scaled, y, test_size=0.7, random_state=0)
    X_cv, X_test, y_cv, y_test = cross_validation.train_test_split(X_test, y_test, test_size=0.5, random_state=0)
    n_cv=len(y_cv)
    n_train=len(y_cv)
    min_cv=np.min(scaler.inverse_transform(X_cv), axis=0)
    max_cv=np.max(scaler.inverse_transform(X_cv), axis=0)

    plot_fname=basename + '_train.png'
    title=base_title + ' training set' + '  ( %d samples )'%n_train
    plot_catalog(scaler.inverse_transform(X_train), y_train, features, labels, title, plot_fname, ranges=(min_cv, max_cv))

    plot_fname=basename + '_cv.png'
    title=base_title + ' cross validation set' + '  ( %d samples )'%n_cv
    plot_catalog(scaler.inverse_transform(X_cv), y_cv, features, labels, title, plot_fname, ranges=(min_cv, max_cv))

    ###########################
    # do a naive bayes analysis
    ###########################
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_cv)

    p,r,f,s = precision_recall_fscore_support(y_cv, y_pred)
    #print "Precision for Naive Bayes : ", p
    #print "Recall for Naive Bayes    : ", r
    print "F1 for Naive Bayes        : ", f
    n_false=np.sum(y_cv != y_pred)
    frac_false=n_false/float(n_cv)
    print "Number of mis-labeled points : %d"% n_false

    X_false = X_cv[y_cv != y_pred]
    y_false = y_cv[y_cv != y_pred]

    plot_fname=basename + '_NB.png'
    title = base_title+' mis-labeled Naive Bayes'+' ( %d / %d   %.2f %% )'%(n_false,n_cv,frac_false*100)
    plot_catalog(scaler.inverse_transform(X_false), y_false, features, labels, title, plot_fname, ranges=(min_cv, max_cv))

    ###########################
    # do a SVM
    ###########################
    svc = svm.SVC(kernel='rbf',C=C, gamma=gamma, probability=True)
    svc.fit(X_train, y_train)
    y_pred=svc.predict(X_cv)
    X_prob=svc.predict_proba(X_cv)

    p,r,f,s = precision_recall_fscore_support(y_cv, y_pred)
    #print "Precision for SVM : ", p
    #print "Recall for SVM    : ", r
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

    #C_range = 10.0 ** np.arange(-2, 9)
    #gamma_range = 10.0 ** np.arange(-5,4)
    #param_grid = dict(gamma=gamma_range, C=C_range)
    #grid = GridSearchCV(svm.SVC(), param_grid=param_grid)
    #grid.fit(X_train, y_train)
    #print("The best classifier is: ", grid.best_estimator_)

