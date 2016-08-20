from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from mlxtend.evaluate import plot_decision_regions

NUM_SHOW = 10

#loading data
iris = load_iris()
X = iris.data[:, [0, 3]]
y = iris.target

#standardize
X[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

lr = LogisticRegression(penalty = 'l2',
						dual = False,
						tol = 0.000001,
						C = 10.0,
						fit_intercept = True,
						intercept_scaling = 1,
						class_weight = None,
						random_state = 1,
						solver = 'newton-cg',
						max_iter = 100,
						multi_class = 'multinomial',
						verbose = 0,
						warm_start = False,
						n_jobs = 1)

lr.fit(X, y)
y_pred = lr.predict(X)
print('Last 5 class labels \n -> Actual: %s  \n Prediciion: %s' %(y[-NUM_SHOW:], y_pred[-NUM_SHOW:]))

plot_decision_regions(X, y, clf=lr)
plt.title('Softmax Regression in scikit_learn')
plt.show()