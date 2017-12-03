import numpy as np
import scipy as sp
from scipy import io
import eigenFaces

import matplotlib.pyplot as plt
from PIL import Image

# computes the accuracy of predictions
def accuracy(y, ypred):
	n = np.size(y)
	n_correct = 0
	for i in range(n):
		if y[i] == ypred[i]:
			n_correct = n_correct + 1
	return n_correct*1.0/n

def main():
        np.random.seed(42)
        # Get training and testing data
        data = sp.io.loadmat('../data/orl_pca.mat')

        # perform PCA
        n, p = np.shape(data['xtrain'])
        plt.ion()
        accuracyList = []
        for k in range(1,200):
                mu, V = eigenFaces.pca(data['xtrain'], k)

                # visualization of the reconstructed images
                xtrain = data['xtrain']
                ytrain = np.squeeze(data['ytrain'])
                xtest = data['xtest']
                ytest = np.squeeze(data['ytest'])

                img = xtrain[1,:].reshape((32, 32),order='F')
                
##                f, axarr = plt.subplots(2,1)
##                axarr[0].set_title('k='+str(k))
##
##                axarr[0].imshow(img)
##
##                z_t = np.dot(V, xtrain[1,:] - mu)
##
##                x_t = np.dot(np.transpose(V), z_t) + mu
##                img_re = x_t.reshape((32, 32),order='F')
##                axarr[1].imshow(img_re, cmap='gray')
##                plt.show()
                


                # use prinicipal directions for predicting eigenFaces
                omega = eigenFaces.centers(mu, V, xtrain, ytrain)
                ytest_pred = eigenFaces.pred(mu, V, omega, xtest)

                # evaluation
                acc = accuracy(ytest, ytest_pred)
                accuracyList.append(acc)
                plt.plot(range(1,k+1),accuracyList)
                plt.pause(0.000001)
                print('Test Accuracy: {0}'.format(acc))

        plt.pause(10)

if __name__ == '__main__':
	main()
