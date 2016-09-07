#!/Users/gopal/projects/learning/tensorflow/venv/bin/python

from sklearn import datasets
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

PLOT_DECISION_BOUNDARIES = True
LOGICAL_XOR = False
PRINT_PERFORMANCE_REPORT = False

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
	np.set_printoptions(suppress=True)  #show in decimal notation and supress scientific notation

	X = iris.data[:, [1,3]]
	y = iris.target

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=0) # 30/70 test/train split

	# normalzing training and test data based on sample mean and variance of the training sample set
	# feature scaling for optimal performance based on gradient descent example
	sc = StandardScaler()
	sc.fit(X_train)
	X_train_std = sc.transform(X_train)
	X_test_std = sc.transform(X_test)

	ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
	ppn.fit(X_train_std, y_train)

	lr = LogisticRegression(C=1000.0, random_state=0)  #C is inverse regularization strength
	lr.fit(X_train_std, y_train)

	svm  = SVC(kernel='linear', C=1.0, random_state=0) #C misclassification penalty, increasing C increases bias and decreases variance
	svm.fit(X_train_std, y_train)

	svm2  = SVC(kernel='rbf', C=1.0, gamma=10.0, random_state=0) #C misclassification penalty, increasing C increases bias and decreases variance
	svm2.fit(X_train_std, y_train)	

	dt = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
	dt.fit(X_train_std, y_train)

	rf = RandomForestClassifier(criterion='entropy', n_estimators=10, random_state=1, n_jobs=2) #n_jobs allow for parallelizing at runtime
	rf.fit(X_train_std, y_train)

	knn = KNeighborsClassifier(n_neighbors = 5, p=2, metric='minkowski') #p=2 is the Euclidean distance
	knn.fit(X_train_std, y_train)

	if PRINT_PERFORMANCE_REPORT:
		y_pred = ppn.predict(X_train_std)
		print
		print "Perceptron Training data performance :"
		print metrics.classification_report(y_train, y_pred)
		y_pred = ppn.predict(X_test_std)
		print
		print "Perceptron Test data performance :"
		print metrics.classification_report(y_test, y_pred)
		y_pred = lr.predict(X_test_std)
		print
		print "Logistic Classifer Test data performance :"
		print metrics.classification_report(y_test, y_pred)

	if LOGICAL_XOR:
		np.random.seed(0)
		X_xor = np.random.randn(200, 2)
		y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:,1] > 0)
		y_xor = np.where(y_xor, 1, -1)

		svm = SVC(kernel = 'rbf', random_state=0, gamma=0.10, C=10.0)
		svm.fit(X_xor, y_xor)


		plt.figure()
		plt.scatter(X_xor[y_xor==1, 0], X_xor[y_xor==1, 1], c='b', marker='x', label='1')
		plt.scatter(X_xor[y_xor==-1, 0], X_xor[y_xor==-1, 1], c='r', marker='o', label='1')		
		plt.ylim(-3.0, 3.0)
		plt.xlim(-3.0, 3.0)
		plt.legend()

		plt.figure()
		plot_decision_regions(X_xor, y_xor, svm)
		plt.title('SVM RBF classifier decision regions and test samples')
		plt.legend(loc='upper left')

		plt.show()

	X_combined_std = np.vstack((X_train_std, X_test_std))
	y_combined_std = np.hstack((y_train, y_test))

	if PLOT_DECISION_BOUNDARIES:
		plt.figure()
		plot_decision_regions(X_combined_std, y_combined_std, ppn, test_idx=range(105, 150))
		plt.title('Perceptron Classfier decision regions and test samples')
		plt.legend(loc='upper left')
		
		plt.figure()
		plot_decision_regions(X_combined_std, y_combined_std, lr, test_idx=range(105, 150))
		plt.title('Logistic Regression Classfier decision regions and test samples')
		plt.legend(loc='upper left')

		plt.figure()
		plot_decision_regions(X_combined_std, y_combined_std, svm, test_idx=range(105, 150))
		plt.title('SVM Linear Classfier decision regions and test samples')
		plt.legend(loc='upper left')

		plt.figure()
		plot_decision_regions(X_combined_std, y_combined_std, svm2, test_idx=range(105, 150))
		plt.title('SVM RBF Classfier decision regions and test samples')
		plt.legend(loc='upper left')

		plt.figure()
		plot_decision_regions(X_combined_std, y_combined_std, dt, test_idx=range(105, 150))
		plt.title('Decision Tree Classfier decision regions and test samples')
		plt.legend(loc='upper left')

		plt.figure()
		plot_decision_regions(X_combined_std, y_combined_std, rf, test_idx=range(105, 150))
		plt.title('Random Forest Classfier decision regions and test samples')
		plt.legend(loc='upper left')

		plt.figure()
		plot_decision_regions(X_combined_std, y_combined_std, knn, test_idx=range(105, 150))
		plt.title('K nearest neighbors Classfier decision regions and test samples')
		plt.legend(loc='upper left')

		plt.show()

	# test_len = X_test_std.shape[0]
	# for n in range(0, test_len):
	# 	print lr.predict_proba(X_test_std[n, :])

	prob = zip(y_test, lr.predict_proba(X_test_std))
	print prob







