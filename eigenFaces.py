import numpy as np
import random
import scipy.linalg as linalg

# INPUTS:
# X - n x p 2d array; where n is the number of images and p is the number of pixels in each image
# k - number of principal directions to compute
# OUTPUTS:
# mu - 1d array of size p; mean vector of rows of X
# V - k x p 2d array; where each row of V is a princiapl direction of X
# The rows of V  should be sorted in the ascending order by eigenvalue.
# **** Note that the order of output matters; use the following order: mu, V ****
def pca(X, k):
##    print(k, X, len(X), len(X[0]))
    V = np.cov(X, rowvar=False)
    Xtrans = np.transpose(X)
    mu = []
    for i in range(len(Xtrans)):
        m = np.mean(Xtrans[i])
        mu.append(m)
##    print(np.shape(V))
    high = len(V) - 1
    eigenvalues,eigenvectors  = linalg.eigh (V, eigvals=(high-k+1, high))
##    eigenvalues,eigenvectors  = linalg.eigh (V, eigvals=(0, 49))
##    print(len(mu))
    
##    print(np.shape(eigenvectors))
##    print(eigenvectors)
    return mu, np.transpose(eigenvectors)



# INPUTS:
# xtrain - nxp 2d array; contains the training images, where n is the number of images and p is the number of pixels in each image
# ytrain - 1d array of size n; contains the labels of training images. 
# Labels range from 0 to 39.
# mu - 1d array of size p; mean vector of rows of xtrain
# V - k x p 2d array; where each row of V is a princiapl direction of X
# OUTPUTS:
# Omega - l x k 2d array; l is the number of classes; where i^th row of Omega contains the center of class i.
def centers(mu, V, xtrain, ytrain):
    L = set(ytrain)
    numk = len(V)
##    print("there are ", len(L), "many classes")
##    print(np.shape(V))
##    print("\n")
    omega = []
    for l in range(len(L)):
        z = np.zeros(numk)
##        print(len(z))
        idx = np.squeeze(np.where(ytrain == l))
        for i in idx:
            j = xtrain[i,:] - mu
            m = (np.dot(V, j))
##            print (np.shape(m))
##            print(m)
            z += m
        z /= len(idx)
        omega.append(z)
    return omega

# INPUTS:
# xtest - nxp 2d array; contains the test images, where n is the number of images and p is the number of pixels in each image 
# mu - 1d array of size p; mean vector of rows of xtrain
# V - k x p 2d array; where each row of V is a princiapl direction of X
# Omega - l x k 2d array; l is the number of classes; where i^th row of Omega contains the center of class i.
# OUTPUTS:
# ypred - 1d array of size n; where ypred[i] is the predicted label of i^th point in xtest.
# Labels range from 0 to 39.
def pred(mu, V, omega, xtest):
##    print(len(xtest))
    ypred = []
    for i in range(len(xtest)):
        z = (np.dot(V, xtest[i,:] - mu))
        distance = float('Inf')
        index = 0
        
        for l in range(len(omega)):
            d = linalg.norm(z-omega[l])
            if d < distance:
                index = l
                distance = d
                
        ypred.append(index)
##        print(len(ypred))
##        print(ypred, np.shape(ypred))
##    print(np.shape(ypred))
    return ypred


		
	
	




