#K-NN classifier for face regonitin


import numpy as np
# functions that may be helpful
from scipy.stats import mode

import sys
#get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time
from helper_functions import loaddata, visualize_knn_2D, visualize_knn_images, plotfaces

print('You\'re running python %s' % sys.version.split(' ')[0])


#The data resides in the files faces.matwhich hold the dataset for further experiments.

# Here, xTr are the training vectors with labels yTr and xTe are the testing vectors with labels yTe

# Visualizing data
# The following script will take the first 10 training images from the face data set and visualize them.


xTr,yTr,xTe,yTe=loaddata("faces.mat")

plt.figure()
plotfaces(xTr[:9, :])



def l2distance(X,Z=None):
    """
    function D=l2distance(X,Z)
    
    Computes the Euclidean distance matrix.
    Syntax:
    D=l2distance(X,Z)
    Input:
    X: nxd data matrix with n vectors (rows) of dimensionality d
    Z: mxd data matrix with m vectors (rows) of dimensionality d
    
    Output:
    Matrix D of size nxm
    D(i,j) is the Euclidean distance of X(i,:) and Z(j,:)
    
    """

    if Z is None:
        Z=X;

    n,d1=X.shape
    m,d2=Z.shape
    assert (d1==d2), "Dimensions of input vectors must match!"


    n,d1=X.shape
    m,d2=Z.shape
    assert (d1==d2), "Dimensions of input vectors must match!"

    #raise NotImplementedError('Your code goes here!')

    S = np.array([np.diag(np.dot(X,X.T)),]*m).transpose()
    R = np.array([np.diag(np.dot(Z,Z.T)),]*n)
    G = np.dot(X,Z.T)
    D = np.sqrt(S + R - 2*G)
    return D





# <p>(b) Implementing the function findknn should 
# find the nearest neighbors of a set of vectors within a given training data set. 


def findknn(xTr,xTe,k):
    """
    function [indices,dists]=findknn(xTr,xTe,k);
    
    Finds the k nearest neighbors of xTe in xTr.
    
    Input:
    xTr = nxd input matrix with n row-vectors of dimensionality d
    xTe = mxd input matrix with m row-vectors of dimensionality d
    k = number of nearest neighbors to be found
    
    Output:
    indices = kxm matrix, where indices(i,j) is the i^th nearest neighbor of xTe(j,:)
    dists = Euclidean distances to the respective nearest neighbors
    """

    K = k
    D_mat = l2distance(xTr,xTe)
    I = D_mat.argsort(axis=0)[0:K,:]

    D_sort = (np.sort(D_mat, axis=0))
    #(np.sort(D_mat.values, axis=0), index=D_mat.index, columns=D_mat.columns)
    D = D_sort[0:K,:]

    return I,D

# The following demo samples random points in 2D. The function should draw direct connections based on where it is clicked
# from the test points to the k  nearest neighbors. 

visualize_knn_2D(findknn) 

visualize_knn_images(findknn, imageType='faces')


# The function analyze computes various metrics to evaluate a classifier. The call of result=analyze(kind,truth,preds);
# outputs the accuracy or absolute loss in variable result. 
# For example, the call
# >> analyze('acc',[1 2 1 2],[1 2 1 1])
# should return an accuracy of 0.75. Here, the true labels are 1,2,1,2 and the predicted labels are 1,2,1,1. 
#  So the first three examples are classified correctly, and the last one is wrong --- 75% accuracy.


def analyze(kind,truth,preds):
    """
    function output=analyze(kind,truth,preds)         
    Analyses the accuracy of a prediction
    Input:
    kind='acc' classification error
    kind='abs' absolute loss
    """
    
    truth = truth.flatten()
    preds = preds.flatten()
    
    if kind == 'abs':
        # compute the absolute difference between truth and predictions
        output = np.mean(np.abs(truth-preds))

    elif kind == 'acc':

        output = (np.sum(truth==preds))/truth.size
    return output


# The function knnclassifier which performs k nearest neighbor classification on a given test data set. 
# The call preds=knnclassifier(xTr,yTr,xTe,k)
# outputs the predictions for the data in xTe



        
def knnclassifier(xTr,yTr,xTe,k):
    """
    function preds=knnclassifier(xTr,yTr,xTe,k);
    
    k-nn classifier 
    
    Input:
    xTr = nxd input matrix with n row-vectors of dimensionality d
    xTe = mxd input matrix with m row-vectors of dimensionality d
    k = number of nearest neighbors to be found
    
    Output:
    
    preds = predicted labels, ie preds(i) is the predicted label of xTe(i,:)
    """
    preds = (mode(yTr[findknn(xTr,xTe,k)[0]], axis=0))[0][0]
    return preds




#  Computing the actual classification error on the test set by calling
# >> analyze("acc",yTe,knnclassifier(xTr,yTr,xTe,3))


print("Face Recognition: (3-nn)")
xTr,yTr,xTe,yTe=loaddata("faces.mat") # load the data
t0 = time.time()
preds = knnclassifier(xTr,yTr,xTe,3)
result=analyze("acc",yTe,preds)
t1 = time.time()
print("You obtained %.2f%% classification acccuracy in %.4f seconds\n" % (result*100.0,t1-t0))




