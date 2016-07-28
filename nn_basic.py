# Adapted from http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/
# $python nn_basic.py

import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib

from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegressionCV
import matplotlib.pyplot as plt

def plot_decision_boundary(pred_func):
	# Set min and max values and give it some padding
	x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
	y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
	h = 0.01
	# Generate a grid of points with distance h between them
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	# Predict the function value for the whole gid
	Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)
	# Plot the contour and training examples
	plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
	plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

# calculating total loss on the dataset
def calculate_loss(model):
	W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
	#forward propagation
	z1 = X.dot(W1) + b1
	a1 = np.tanh(z1)
	z2 = a1.dot(W2) + b2
	#computing softmax
	exp_scores = np.exp(z2)
	probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
	correct_logprobs = -np.log(probs[range(num_examples), y])
	data_loss = np.sum(correct_logprobs)
	#regularization term
	data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
	return 1./num_examples * data_loss

def predict(model, x):
	W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
	#forward propagation
	z1 = x.dot(W1) + b1
	a1 = np.tanh(z1)
	z2 = a1.dot(W2) + b2
	#computing softmax
	exp_scores = np.exp(z2)
	probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
	# returns index of class with larger output from softmax
	return np.argmax(probs, axis=1)	

# Learns parameters and returns model
# nn_hdim: Number of nodes in hidden layer
# num_passes: Number of passes through training data set for gradient descent
# print_loss: If True, print loss every 1000 iterations
def build_model(nn_hdim, num_passes=20000, print_loss=False):

	#Intializing parameters
	np.random.seed(0)
	W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
	b1 = np.zeros((1, nn_hdim))
	W2 = np.random.rand(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
	b2 = np.zeros((1, nn_output_dim))

	model  = {}

	#Gradient Descent for each batch
	for i in xrange(0, num_passes):
		#forward propagation
		z1 = X.dot(W1) + b1
		a1 = np.tanh(z1)
		z2 = a1.dot(W2) + b2
		#computing softmax
		exp_scores = np.exp(z2)
		probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

		#backpropagation
		delta3 = probs
		delta3[range(num_examples), y] -= 1
		dW2 = (a1.T).dot(delta3)
		db2 = np.sum(delta3, axis=0, keepdims=True)
		delta2 = delta3.dot(W2.T)*(1-np.power(a1, 2))
		dW1 = np.dot(X.T, delta2)
		db1 = np.sum(delta2, axis=0)

		#regularization terms
		dW2 += reg_lambda * W2
		dW1 += reg_lambda * W1

		#parameter update
		W1 += -epsilon * dW1
		W2 += -epsilon * dW2
		b1 += -epsilon * db1		
		b2 += -epsilon * db2

		model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

		if print_loss and i%1000 == 0:
			print "Loss after iteration %i: %f" %(i, calculate_loss(model))

	return model


# input data param
noise_coef = 0.15 #noise term
num_in_samples = 2000

# parameters for gradient descent
num_examples = num_in_samples
nn_input_dim = 2
nn_output_dim = 2
nn_hidden_dim = 50
epsilon = 0.01 #learning rate
reg_lambda = 0.01 #regularization strength

X, y = sklearn.datasets.make_moons(num_in_samples, noise=noise_coef)

clf = LogisticRegressionCV()
clf.fit(X, y)

model = build_model(nn_hidden_dim, print_loss=True)

plt.scatter(X[:, 0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
plot_decision_boundary(lambda x: predict(model, x))
#plot_decision_boundary(lambda x: clf.predict(x))
plt.show()




