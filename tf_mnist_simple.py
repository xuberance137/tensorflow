#!/Users/gopal/projects/learning/tensorflow/venv/bin/python

import tensorflow as tf
from tqdm import tqdm
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

N = 28
Nin = N*N
Nout = 10
batch_size = 100

x = tf.placeholder(tf.float32, [None, Nin])
W = tf.Variable(tf.zeros([Nin, Nout]))
b = tf.Variable(tf.zeros([Nout]))

y = tf.nn.softmax(tf.matmul(x, W) + b) #predicted labels
y_ = tf.placeholder(tf.float32, [None, Nout]) #correct labels

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for n in tqdm(range(10000)):
	batch_xs, batch_ys = mnist.train.next_batch(batch_size)
	sess.run(train_step, feed_dict = {x : batch_xs, y_ : batch_ys})

correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

print sess.run(accuracy, feed_dict = {x:mnist.test.images, y_:mnist.test.labels})

