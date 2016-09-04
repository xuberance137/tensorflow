#!/Users/gopal/projects/learning/tensorflow/venv/bin/python

from sklearn import datasets
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron

import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def plot_decision_regions(X, y, classifier, test_idx = None, resolution = 0.02):
	#setup marker generation and color map
	markers = ('s', 'x', 'o', '^', 'v')
	colors = ('lightgreen', 'red', 'cyan', 'blue',  'gray')
	cmap = ListedColormap(colors[:len(np.unique(y))])

	#plot the decison surface
	x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1	

	xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
	Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
	Z = Z.reshape(xx1.shape)

	plt.contourf(xx1, xx2, Z, alpha = 0.4, cmap = cmap)
	plt.xlim(xx1.min(), xx1.max())
	plt.ylim(xx2.min(), xx2.max())

	#plot class samples
	for idx, c1 in enumerate(np.unique(y)):
		plt.scatter(x = X[y==c1, 0], y = X[y==c1, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=c1)

	#highlight test samples
	if test_idx:
		X_test, y_test = X[test_idx, :], y[test_idx]
		plt.scatter(X_test[:, 0], X_test[:, 1], c='', alpha=1.0, linewidths=1, marker='o', s=55, label='test set')


if __name__ == '__main__':

	iris = datasets.load_iris()

	X = iris.data[:, [2,3]]
	y = iris.target

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) # 30/70 test/train split

	sc = StandardScaler()
	sc.fit(X_train)
	X_train_std = sc.transform(X_train)
	X_test_std = sc.transform(X_test)

	ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
	ppn.fit(X_train_std, y_train)

	y_pred = ppn.predict(X_train_std)
	print
	print "Training data performance :"
	print metrics.classification_report(y_train, y_pred)
	y_pred = ppn.predict(X_test_std)
	print
	print "Test data performance :"
	print metrics.classification_report(y_test, y_pred)

	X_combined_std = np.vstack((X_train_std, X_test_std))
	y_combined_std = np.hstack((y_train, y_test))

	plot_decision_regions(X_combined_std, y_combined_std, ppn, test_idx=range(105, 150))
	plt.title('Classfier decision regions and test samples')
	plt.legend(loc='upper left')
	plt.show()












