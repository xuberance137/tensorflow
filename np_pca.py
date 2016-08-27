"""
http://sebastianraschka.com/Articles/2014_pca_step_by_step.html
"""

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

PRINT_INTERMEDIATE_OUTPUTS = True
PLOT_DATA_SAMPLES = True
sample_size = 20

np.random.seed(11212121) # so that we get the same plot every run

def plot3d(class1_sample, class2_sample):
	fig = plt.figure(figsize=(8,8))
	ax = fig.add_subplot(111, projection='3d')
	plt.rcParams['legend.fontsize'] = 10   
	ax.plot(class1_sample[0,:], class1_sample[1,:], class1_sample[2,:], 'o', markersize=8, color='blue', alpha=0.5, label='class1')
	ax.plot(class2_sample[0,:], class2_sample[1,:], class2_sample[2,:], '^', markersize=8, alpha=0.5, color='green', label='class2')

	plt.title('Samples for class 1 and class 2')
	ax.legend(loc='upper right')

def plot2d(transformed):
	fig = plt.figure(figsize=(8,8))
	plt.plot(transformed[0, 0:sample_size], transformed[1, 0:sample_size], 'o', markersize=7, color='blue', alpha=0.5, label='class1')
	plt.plot(transformed[0, sample_size:2*sample_size], transformed[1, sample_size:2*sample_size], '^', markersize=7, color='green', alpha=0.5, label='class2')
	plt.xlim([-4,4])
	plt.ylim([-4,4])
	plt.xlabel('x_values')
	plt.ylabel('y_values')
	plt.legend()
	plt.title('Transformed samples with class labels')

# creating 40 3min samples randomly drawn from a multivariate gaussian distribution
mu1 = np.array([0,0,0])
mu2 = np.array([1,1,1])
cov_mat1 = np.array([[1,0,0],[0,1,0],[0,0,1]])
cov_mat2 = np.array([[1,0,0],[0,1,0],[0,0,1]])
class1_sample = np.random.multivariate_normal(mu1, cov_mat1, sample_size).T
class2_sample = np.random.multivariate_normal(mu2, cov_mat2, sample_size).T
# concatenate data sets for PCA
all_samples = np.concatenate((class1_sample, class2_sample), axis=1)

mean_x = np.mean(all_samples[0, :])
mean_y = np.mean(all_samples[1, :])
mean_z = np.mean(all_samples[2, :])
mean_vec = np.array([mean_x, mean_y, mean_z])

scatter_mat = np.zeros((mu1.shape[0], mu1.shape[0]))
# sum(x_k - m).(x_k - m)T
for n in range(all_samples.shape[1]):
	scatter_mat += (all_samples[:, n].reshape(mu1.shape[0], 1) - mean_vec).dot((all_samples[:, n].reshape(mu1.shape[0], 1) - mean_vec))

cov_mat  = np.cov([all_samples[0,:], all_samples[1,:], all_samples[2, :]])

# eigen vectors and eigen vals from scatter matrix
eig_val_sc , eig_vec_sc = np.linalg.eig(scatter_mat)
eig_val_cov , eig_vec_cov = np.linalg.eig(cov_mat)

# checking if noramlized all eigengects to have unit length -> important
# for ev in eig_vec_sc:
#     np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))

# creating pairs of (eigen_values, eigen_vectors) tuples
eig_pairs = [(np.abs(eig_val_sc[i]), eig_vec_sc[:,i]) for i in range(len(eig_val_sc))]
#sort in descending order based on eigen values
eig_pairs.sort(key=lambda x: x[0], reverse=True)
# reducing 3D feature space to 2D feature space
#stacking arrays in sequence horizontally
matrix_w = np.hstack((eig_pairs[0][1].reshape(3,1), eig_pairs[1][1].reshape(3,1)))

transformed_samples = matrix_w.T.dot(all_samples)

if PRINT_INTERMEDIATE_OUTPUTS:
	print
	print class1_sample.shape
	print class2_sample.shape
	print all_samples.shape
	print mean_vec
	print scatter_mat
	print cov_mat
	print
	print eig_val_sc
	print eig_vec_sc
	print
	print eig_val_cov
	print eig_vec_cov
	print
	print eig_pairs
	print
	print matrix_w
	print matrix_w.T.shape	
	print
	print transformed_samples

if PLOT_DATA_SAMPLES:
	plot3d(class1_sample, class2_sample)
	plot2d(transformed_samples)
	plt.show()





