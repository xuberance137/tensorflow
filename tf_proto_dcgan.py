#!/Users/gopal/projects/learning/tensorflow/venv/bin/python
"""
Adapted from http://blog.evjang.com/2016/06/generative-adversarial-nets-in.html
Basic Equations:

x -> D1 -> D1(x)
z -> G -> x'
z -> G -> x' -> D2 -> D2(G(z))

For ideal discrimination, we want to maximize output of D1 and minimize output of D2
Generator is designed to maximize output of D2, ie likelihood of input belonging to p_data
D1 and D2 are copies of D ie they are networks sharing the same parameters
value function = log D1(x) + log (1-D2(G(z)))
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from tqdm import tqdm

SHOW_TRAINING_DECISION_BOUNDARIES = False
mu, sigma = -1, 1
TRAIN_ITER = 8000
M = 200 # minbatch size
# xs = np.linspace(-5, 5, 1000)
# #plt.plot(xs, norm.pdf(xs, loc=mu, scale=signma))
# sns.regplot(xs, norm.pdf(xs, loc=mu, scale=signma), fit_reg=False, color="g")
# plt.show()

# plot decision surface
def plot_d0(D,input_node, iter):
	f=plt.figure(iter)
	ax = f.add_subplot(1,1,1)
	# p_data
	xs=np.linspace(-5,5,1000)
	ax.plot(xs, norm.pdf(xs,loc=mu,scale=sigma), label='p_data')
	# decision boundary
	r=1000 # resolution (number of points)
	xs=np.linspace(-5,5,r)
	ds=np.zeros((r,1)) # decision surface
	# process multiple points in parallel in a minibatch
	for i in range(r/M):
		x=np.reshape(xs[M*i:M*(i+1)],(M,1))
		ds[M*i:M*(i+1)]=sess.run(D,{input_node: x})

	ax.plot(xs, ds, label='decision boundary')
	ax.set_ylim(0,1.1)
	plt.legend()
	plt.title("Evolving Decision Boundary at %d interations" %iter)


def plot_fig(iter):
	# plots pg, pdata, decision boundary 
	f=plt.figure(iter)
	ax = f.add_subplot(1,1,1)
	# p_data
	xs=np.linspace(-5,5,1000)
	ax.plot(xs, norm.pdf(xs,loc=mu,scale=sigma), label='p_data')

	# decision boundary
	r=5000 # resolution (number of points)
	xs=np.linspace(-5,5,r)
	ds=np.zeros((r,1)) # decision surface
	# process multiple points in parallel in same minibatch
	for i in range(r/M):
		x=np.reshape(xs[M*i:M*(i+1)],(M,1))
		ds[M*i:M*(i+1)]=sess.run(D1,{x_node: x})

	ax.plot(xs, ds, label='decision boundary')

	# distribution of inverse-mapped points
	zs=np.linspace(-5,5,r)
	gs=np.zeros((r,1)) # generator function
	for i in range(r/M):
		z=np.reshape(zs[M*i:M*(i+1)],(M,1))
		gs[M*i:M*(i+1)]=sess.run(G,{z_node: z})
	histc, edges = np.histogram(gs, bins = 10)
	ax.plot(np.linspace(-5,5,10), histc/float(r), label='p_g')

	# ylim, legend
	ax.set_ylim(0,1.1)
	plt.legend()


# Multilayer perceptron
def mlp(input, output_dim):
	# initialize learning parameters
	w1 = tf.get_variable("w0", [input.get_shape()[1], 6], initializer=tf.random_normal_initializer())
	b1 = tf.get_variable("b0", [6], initializer=tf.constant_initializer(0.0))
	w2 = tf.get_variable("w1", [6, 5], initializer=tf.random_normal_initializer())
	b2 = tf.get_variable("b1", [5], initializer=tf.constant_initializer(0.0))
	w3 = tf.get_variable("w2", [5, output_dim], initializer=tf.random_normal_initializer())
	b3 = tf.get_variable("b2", [output_dim], initializer=tf.constant_initializer(0.0))	
	# NN operators
	fc1 = tf.nn.tanh(tf.matmul(input, w1) + b1)
	fc2 = tf.nn.tanh(tf.matmul(fc1, w2) + b2)
	fc3 = tf.nn.tanh(tf.matmul(fc2, w3) + b3)

	return fc3, [w1, b1, w2, b2, w3, b3]

def momentum_optimizer(loss, var_list):
	batch = tf.Variable(0)
	learning_rate = tf.train.exponential_decay(0.001, batch, TRAIN_ITER//4, 0.95, staircase=True)
	optimizer = tf.train.MomentumOptimizer(learning_rate, 0.6).minimize(loss, global_step=batch, var_list=var_list)

	return optimizer

with tf.variable_scope("D_pre"):
	input_node = tf.placeholder(tf.float32, shape=(M, 1))
	train_labels = tf.placeholder(tf.float32, shape=(M,1))
	D, theta = mlp(input_node, 1)
	loss = tf.reduce_mean(tf.square(D-train_labels)) #MSE

optimizer = momentum_optimizer(loss, None)
sess = tf.InteractiveSession()
tf.initialize_all_variables().run()

N=0
if SHOW_TRAINING_DECISION_BOUNDARIES:
	plot_d0(D, input_node, N)

N=100
lh = np.zeros(N)
for n in range(N):
	d = (np.random.random(M)-0.5)*10.0 #covering the domain [-5, 5]
	labels = norm.pdf(d, loc=mu, scale=sigma)
	lh[n], _  = sess.run([loss, optimizer], {input_node: np.reshape(d, (M,1)), train_labels:np.reshape(labels, (M,1))})

weigthsD = sess.run(theta) #run session to get learned weights and store in temp array
if SHOW_TRAINING_DECISION_BOUNDARIES:
	plot_d0(D, input_node, N)

N=1000
lh = np.zeros(N)
for n in range(N):
	d = (np.random.random(M)-0.5)*10.0 #covering the domain [-5, 5]
	labels = norm.pdf(d, loc=mu, scale=sigma)
	lh[n], _  = sess.run([loss, optimizer], {input_node: np.reshape(d, (M,1)), train_labels:np.reshape(labels, (M,1))})

weigthsD = sess.run(theta) #run session to get learned weights and store in temp array

if SHOW_TRAINING_DECISION_BOUNDARIES:
	plot_d0(D, input_node, N)

sess.close()
# plt.show() #to show all the decision boundary plots

with tf.variable_scope("G"):
	z_node = tf.placeholder(tf.float32, shape=(M, 1))  #M uniform floats
	G, theta_g = mlp(z_node, 1)
	G = tf.mul(5.0, G) #scale by 5 to match range

with tf.variable_scope("D") as scope:
	x_node = tf.placeholder(tf.float32, shape=(M, 1)) #normally distributed floats
	fc, theta_d = mlp(x_node, 1)
	D1 = tf.maximum(tf.minimum(fc, 0.99), 0.01) #clamp output to the range [0.01, 0.99]
	scope.reuse_variables()
	fc, theta_d  = mlp(G, 1)
	D2 = tf.maximum(tf.minimum(fc, 0.99), 0.01)

obj_d = tf.reduce_mean(tf.log(D1) + tf.log(1 - D2)) #goal of D is to maximize D1(x) and D2(G(z))
obj_g = tf.reduce_mean(tf.log(D2))  #goal of G is to maximize D(G(z))

opt_d = momentum_optimizer(1-obj_d, theta_d)
opt_g = momentum_optimizer(1-obj_g, theta_g) #maximize each

sess = tf.InteractiveSession()
tf.initialize_all_variables().run()

#copy weights from pretraining to new D network
for i,v in enumerate(theta_d):
	sess.run(v.assign(weigthsD[i]))

plot_fig(1)
plt.title('Before Training')

k = 1
histd, histg = np.zeros(TRAIN_ITER), np.zeros(TRAIN_ITER)

for i in tqdm(range(TRAIN_ITER)):
	for j in range(k):
		x = np.random.normal(mu, sigma, M)
		x.sort()
		z = np.linspace(-5.0, 5.0, M) + np.random.random(M)*0.01 #sample m-batch from noise prior
		histd[i], _ = sess.run([obj_d, opt_d], {x_node: np.reshape(x, (M, 1)), z_node: np.reshape(z, (M, 1))}) #updaet discriminator
		
	z = np.linspace(-5.0, 5.0, M) + np.random.random(M)*0.01 #sample noise prior
	histg[i], _ = sess.run([obj_g, opt_g], {x_node: np.reshape(x, (M, 1)), z_node: np.reshape(z, (M, 1))}) #update generator


plot_fig(2)
plt.title('After Training')


# plt.plot(range(TRAIN_ITER),histd, label='obj_d')
# plt.plot(range(TRAIN_ITER), 1-histg, label='obj_g')
# plt.legend()
plt.show()
























