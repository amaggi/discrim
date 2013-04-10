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
from ml import NeuralNetwork
from helpers import random_weights

#################
# create datasets
#################

# Trimouns
#filename='trimouns.hdf5'
#eq_cat=generate_eq_catalog(1000,-30,30,-30,30)
#blast_cat=generate_blast_catalog(1000,0,0,3,3,30,17,0.1)
#write_syn_catalogs_hdf5(eq_cat, blast_cat, filename)
#X, y, features, labels = read_catalogs_hdf5(filename)
#plot_catalogs(X, y, features, labels, 'Trimouns synthetic catalog', 'trimouns_syn.pdf')


# Mollard
#filename='mollard.hdf5'
#eq_cat=generate_eq_catalog(1000,-40,40,-40,40)
#blast_cat=generate_blast_catalog(1000,0,0,10,10,30,12,2)
#write_syn_catalogs_hdf5(eq_cat, blast_cat, filename)
#X, y, features, labels = read_catalogs_hdf5(filename)
#plot_catalogs(X, y, features, labels, 'Mollard synthetic catalog', 'mollard_syn.pdf')

# skew
#filename='skew.hdf5'
#eq_cat=generate_eq_catalog(1000,-40,40,-40,40)
#blast_cat=generate_blast_catalog(1000,0,0,5,15,30,12,2)
#write_syn_catalogs_hdf5(eq_cat, blast_cat, filename)
#X, y, features, labels = read_catalogs_hdf5(filename)
#plot_catalogs(X, y, features, labels, 'Skew synthetic catalog', 'skew_syn.pdf')

#exit()

###########################
# read and pre-process data
###########################

# eq = 0
# blast = 1

filename='skew.hdf5'
X, y, features, labels = read_catalogs_hdf5(filename)

# do feature scaling
X_scaled = preprocessing.scale(X)

# do PCA
#pca = PCA(3)
#pca.fit(X_scaled)
#print pca.explained_variance_ratio_

# split out a training set
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_scaled, y, test_size=0.7, random_state=0)
X_cv, X_test, y_cv, y_test = cross_validation.train_test_split(X_test, y_test, test_size=0.5, random_state=0)

print filename
###########################
# do a naive bayes analysis
###########################
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_cv)

p,r,f,s = precision_recall_fscore_support(y_cv, y_pred)
#print "Precision for Naive Bayes : ", p
#print "Recall for Naive Bayes    : ", r
print "F1 for Naive Bayes        : ", f
print "Number of mis-labeled points : %d"% np.sum(y_cv != y_pred)

###########################
# do a SVM
###########################
svc = svm.SVC(kernel='rbf',C=100.0, gamma=0.1)
svc.fit(X_train, y_train)
y_pred=svc.predict(X_cv)

p,r,f,s = precision_recall_fscore_support(y_cv, y_pred)
#print "Precision for SVM : ", p
#print "Recall for SVM    : ", r
print "F1 for SVM        : ", f
print "Number of mis-labeled points : %d"% np.sum(y_cv != y_pred)

#C_range = 10.0 ** np.arange(-2, 9)
#gamma_range = 10.0 ** np.arange(-5,4)
#param_grid = dict(gamma=gamma_range, C=C_range)
#grid = GridSearchCV(svm.SVC(), param_grid=param_grid)
#grid.fit(X_cv, y_cv)
#print("The best classifier is: ", grid.best_estimator_)

###########################
# try a neural network
###########################
l1_shape = (14,7)
l2_shape = (2,15)
theta1 = random_weights(l1_shape)
theta2 = random_weights(l2_shape)
thetas = (theta1, theta2)
lambda_value = 2

nn = NeuralNetwork(thetas, training_set = X_train, labels = y_train)
nn.train(lambda_value, maxiter=10, disp=False)
y_pred = np.array([ np.argmax(nn.predict(vec)) for vec in X_cv])

p,r,f,s = precision_recall_fscore_support(y_cv, y_pred)
print "Fit for Neural Network : ", f
print "Number of mis-labeled points : %d"% np.sum(y_cv != y_pred)


