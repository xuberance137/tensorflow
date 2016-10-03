#!/Users/gopal/projects/learning/tensorflow/venv/bin/python

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# class for sequential backward selection algorithm
class SBS():

	def __init__(self, estimator, k_features, scoring=accuracy_score, test_size=0.25, random_state=1):
		self.scoring = scoring
		self.estimator = clone(estimator) #creates a deep copy of the estimator with same params but not fit on any data
		self.k_features = k_features
		self.test_size = test_size
		self.random_state = random_state

	def fit(self, X, y):
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
		dim = X_train.shape[1]
		self.indices_ = tuple(range(dim))
		self.subsets_ = [self.indices_]
		score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)
		self.scores_ = [score]

		while dim > self.k_features:
			scores = []
			subsets = []

			for p in combinations(self.indices_, r=dim-1):  #unique combinations with dim-1 values
				score = self._calc_score(X_train, y_train, X_test, y_test, p)
				scores.append(score)
				subsets.append(p)
				print p

			best = np.argmax(scores)
			self.indices_ = subsets[best]
			print "Best : ", self.indices_
			self.subsets_.append(self.indices_)
			dim -= 1  #iterating down from dim to k+1 which would look at combinations from dim-1 to k
			self.scores_.append(scores[best])

		self.k_score_ = self.scores_[-1] # last value is that for k dimensions

		return self

	def transform(self, X):
		return X[:, self.indices_] # self.indices_ has the k dim values in the last itneration for p

	def _calc_score(self, X_train, y_train, X_test, y_test, indices):
		self.estimator.fit(X_train[:, indices], y_train)
		y_pred = self.estimator.predict(X_test[:, indices])
		score = self.scoring(y_test, y_pred)
		return score


if __name__ == '__main__':

	df_wine = pd.read_csv('data/winedata/wine.data', header=None)
	df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Phenols', 'Flavanoids', 'Non-Flav Phenols', 'Proanthocyanins', 'Color Intensity', 'Hue', 'OD of diluted wines', 'Proline']

	#print df_wine.head()

	X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.3, random_state = 0)

	mms = MinMaxScaler()
	X_train_norm = mms.fit_transform(X_train)
	X_test_norm = mms.transform(X_test)
	stdsc = StandardScaler()
	X_train_std = stdsc.fit_transform(X_train)
	X_test = stdsc.transform(X_test)

	lr = LogisticRegression(penalty='l1', C =0.1)
	lr.fit(X_train_std, y_train)
	#print lr.coef_

	knn = KNeighborsClassifier(n_neighbors=2)
	sbs = SBS(knn, k_features=1)
	sbs.fit(X_train_std, y_train)

	k_feat = [len(k) for k in sbs.subsets_]
	plt.plot(k_feat, sbs.scores_, marker='o')
	plt.ylim([0.7, 1.1])
	plt.ylabel('Accuracy')
	plt.xlabel('Number of features')
	plt.grid()
	#plt.show()

	feat_labels = df_wine.columns[1:]

	forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
	forest.fit(X_train, y_train)
	importances = forest.feature_importances_
	indices = np.argsort(importances)[::-1] #descending order of values in importances
	for f in range(X_train.shape[1]):
		print importances[indices[f]], "\t", feat_labels[indices[f]]

	plt.figure()
	plt.title('Feature Importances')
	plt.bar(range(X_train.shape[1]), importances[indices], color='lightblue', align='center')
	plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation=90)
	plt.xlim([-1, X_train.shape[1]])
	plt.tight_layout()
	plt.show()








