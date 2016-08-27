"""
Adapted from
https://lazyprogrammer.me/bayes-classifier-and-naive-bayes-tutorial-using/
Original code at
https://github.com/lazyprogrammer/machine_learning_examples/blob/master/bayes.py
Run as 
$python np_naive_bayes.py reconstruct

classifier output category C ~ P(C/X) where X is the data
P(C/X) is the posterior
P(X/C) is the likelihood
Bayes Rule : P(C/X) =  P(X/C)*P(X)/P(C)
Assuming uniform sampling in test data, P(C) for MNIST digit data is 1/10
P(X) is constant if constant for every category
Hence, P(C/X) ~ P(X/C)
C = argmax_c P(X/C)

Likelihood can be derived from the multivariate Gaussian:
fx(x1,x2,x3...,xk) = 1/srqt(2*pi(det(cov(X)))) * exp(-0.5*(X-mu)T.cov(X)-1.(X-mu))

instead of max likelihood as above, we can calculate the log likelihood

ln(L) = -0.5ln(det(cov(X))) -0.5*k*ln(2*pi) -0.5*(X-mu)T.cov(X)-1.(X-mu)

Naive Bayes assumes that all input features in X =(x1,x2,x3,...) are independent

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from IPython.core.pylabtools import figsize
import pandas as pd
import sys

PLOT_POISSON_DIST = False

figsize(12.5, 4)

a = np.arange(24)
poi = stats.poisson
lambda_ = [1.5, 2.5, 3.5]
colours = ["#348ABD", "#A60628", "#FF00FF"]

if PLOT_POISSON_DIST:
	plt.bar(a, poi.pmf(a, lambda_[0]), color=colours[0], label="$\lambda = %.1f$" % lambda_[0], alpha=0.60, edgecolor=colours[0], lw="3")
	plt.bar(a, poi.pmf(a, lambda_[1]), color=colours[1], label="$\lambda = %.1f$" % lambda_[1], alpha=0.60, edgecolor=colours[1], lw="3")
	plt.bar(a, poi.pmf(a, lambda_[2]), color=colours[2], label="$\lambda = %.1f$" % lambda_[2], alpha=0.60, edgecolor=colours[2], lw="3")
	plt.xticks(a + 0.4, a)
	plt.legend()
	plt.ylabel("probability of $k$")
	plt.xlabel("$k$")
	plt.title("Probability mass function of a Poisson random variable; differing $\lambda$ values")
	plt.show()


Xtest = pd.read_csv("data/mnist_csv/Xtest.txt", header=None)
Xtrain = pd.read_csv("data/mnist_csv/Xtrain.txt", header=None)
Ytest = pd.read_csv("data/mnist_csv/label_test.txt", header=None)
Ytrain = pd.read_csv("data/mnist_csv/label_train.txt", header=None)

class Bayes(object):
	def fit(self, X, y):
		self.gaussians = dict()  #gaussians is a dict with 10(number digits) entries with 20 mu values and 20*20 cov matrix 
		labels = y.as_matrix().flatten()
		for index in range(len(labels)):
			c = labels[index]
			current_x = Xtrain[Ytrain[0]==c]
			self.gaussians[c] = {
				'mu' : current_x.mean(),
				'sigma' : np.cov(current_x.T)
			}
			# if index == 0:
			# 	plt.imshow(self.gaussians[c]['sigma'])
			# 	plt.show()

	def distributions(self, x):
		lls = np.zeros(len(self.gaussians))
		for c, g in self.gaussians.iteritems():
			x_minus_mu = x - g['mu']
			k1 = np.log(2*np.pi)*x.shape + np.log(np.linalg.det(g['sigma']))
			k2 = np.dot( np.dot(x_minus_mu, np.linalg.inv(g['sigma'])), x_minus_mu)
			ll = -0.5 * (k1 + k2)
			lls[c] = ll

		return lls

	def predict_one(self, x):
		lls = self.distributions(x)
		return np.argmax(lls)

	def predict(self, X):
		Ypred = X.apply(lambda x: self.predict_one(x), axis=1)
		return Ypred


if __name__ == '__main__':
	bayes = Bayes()
	bayes.fit(Xtrain, Ytrain)
	Ypred = bayes.predict(Xtest)
	#confusion matrix
	C = np.zeros((10, 10), dtype=np.int)
	for p , t in zip(Ypred.as_matrix().flatten(), Ytest.as_matrix().flatten()):
		C[t, p] += 1
	print
	print "Confusion Matrix :"
	print
	print C
	print
	print "Accuracy : ", np.trace(C) / float(Ytest.size)

	if 'reconstruct' in sys.argv:
		# show means as images
		Q = pd.read_csv("data/mnist_csv/Q.txt", header=None).as_matrix()
		for c,g in bayes.gaussians.iteritems():
			y = np.dot(Q, g['mu'].as_matrix())
			y = np.reshape(y, (28,28))
			if c == 1:
				plt.imshow(y)
				plt.title(c)
				plt.show()

















