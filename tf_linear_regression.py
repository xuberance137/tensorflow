
import numpy as np
import seaborn
import tensorflow as tf
import matplotlib.pyplot as plt

# parameters
n_samples = 1000
n_steps = 25000
batch_size = 100
# sample data
X_data = np.arange(100, step=0.1)
y_data = X_data + 20*np.sin(X_data/10.0)
# resize as TF is specific on shapes
X_data = np.reshape(X_data, (n_samples, 1))
y_data = np.reshape(y_data, (n_samples, 1))
# define placeholders for input
X = tf.placeholder(tf.float32, shape=(None, 1))
y = tf.placeholder(tf.float32, shape=(None, 1))
# learning variables
with tf.variable_scope("linear-regression"):
    W = tf.get_variable("w", (1,1), initializer=tf.random_normal_initializer())
    b = tf.get_variable("b", (1,), initializer=tf.constant_initializer(0.0))
    y_pred = tf.matmul(X, W) + b
    loss = tf.reduce_sum((y-y_pred)**2/n_samples)
# running gradient descent
opt_operation = tf.train.AdamOptimizer().minimize(loss)
y_pred_val = []
loss_val = []
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    # gradient descent loop for n_steps
    for k in range(n_steps):
        # select minibatch
        indices = np.random.choice(n_samples, batch_size)
        X_batch, y_batch = X_data[indices], y_data[indices]
        k, loss_v, y_pred_v = sess.run([opt_operation, loss, y_pred], feed_dict={X:X_batch, y:y_batch})
    # plotting preduction results
    k, loss_val, y_pred_val = sess.run([opt_operation, loss, y_pred], feed_dict={X:X_data, y:y_data})
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(X_data, y_data, c='b', label='true')
ax.scatter(X_data, y_pred_val, c='r', label='predicted') # should be a straight line because we are curve fitting with one parameter
plt.legend(loc='upper left')
plt.show()


