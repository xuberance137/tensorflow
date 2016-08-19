# documented at : http://bcomposes.com/2015/11/26/simple-end-to-end-tensorflow-examples/ 
# adapted from : https://github.com/jasonbaldridge/try-tf/blob/master/softmax.py
# data from : https://github.com/jasonbaldridge/try-tf/tree/master/simdata
# run as : $python tf_simplemodel.py --train data/saturn_data_train.csv --test data/saturn_data_eval.csv --num_epochs 100 --verbose True

import tensorflow.python.platform

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Global variables.
NUM_LABELS = 2    # The number of labels.
BATCH_SIZE = 100  # The number of training examples to use per training step.

# Define the flags useable from the command line.
tf.app.flags.DEFINE_string('train', None, 'File containing the training data (labels & features).')
tf.app.flags.DEFINE_string('test', None, 'File containing the test data (labels & features).')
tf.app.flags.DEFINE_integer('num_epochs', 1, 'Number of examples to separate from the training data for the validation set.')
tf.app.flags.DEFINE_integer('num_hidden', 1, 'Number of nodes in the hidden layer.')
tf.app.flags.DEFINE_boolean('verbose', False, 'Produce verbose output.')
tf.app.flags.DEFINE_boolean('hidden_layer', False, 'Produce either logistic regression(FALSE) or neural network with num_hidden layers (TRUE).')
FLAGS = tf.app.flags.FLAGS

# Extract numpy representations of the labels and features given rows consisting of:
#   label, feat_0, feat_1, ..., feat_n
def extract_data(filename, plot_data=False):

    # Arrays to hold the labels and feature vectors.
    labels = []
    fvecs = []

    # Iterate over the rows, splitting the label from the features. Convert labels
    # to integers and features to floats.
    for line in file(filename):
        row = line.split(",")
        labels.append(int(row[0]))
        fvecs.append([float(x) for x in row[1:]])

    # Convert the array of float arrays into a numpy float matrix.
    fvecs_np = np.matrix(fvecs).astype(np.float32)
    # Convert the array of int labels into a numpy array, uint8 works because it is a binary output.
    labels_np = np.array(labels).astype(dtype=np.uint8)
    # Convert the int numpy array into a one-hot matrix.
    labels_onehot = (np.arange(NUM_LABELS) == labels_np[:, None]).astype(np.float32)

    if plot_data:
        plt.scatter(fvecs_np[:, 0], fvecs_np[:,1], s=40, c=labels_np, cmap=plt.cm.Spectral)
        #plot_decision_boundary(lambda x: predict(model, x))
        #plot_decision_boundary(lambda x: clf.predict(x))
        plt.show()

    # Return a pair of the feature matrix and the one-hot label matrix.
    return fvecs_np,labels_onehot


# Init weights method. (Lifted from Delip Rao: http://deliprao.com/archives/100)
def init_weights(shape, init_method='xavier', xavier_params = (None, None)):
    if init_method == 'zeros':
        return tf.Variable(tf.zeros(shape, dtype=tf.float32))
    elif init_method == 'uniform':
        return tf.Variable(tf.random_normal(shape, stddev=0.01, dtype=tf.float32))
    else: #xavier
        (fan_in, fan_out) = xavier_params
        low = -4*np.sqrt(6.0/(fan_in + fan_out)) # {sigmoid:4, tanh:1} 
        high = 4*np.sqrt(6.0/(fan_in + fan_out))
        return tf.Variable(tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32))
    

def main(argv=None):

    # Be verbose?
    verbose = FLAGS.verbose
    
    # Get the data.
    train_data_filename = FLAGS.train
    test_data_filename = FLAGS.test

    # Extract it into numpy matrices.
    train_data, train_labels = extract_data(train_data_filename)
    test_data, test_labels = extract_data(test_data_filename, True)

    # Get the shape of the training data.
    train_size,num_features = train_data.shape

    # Get the number of epochs for training.
    num_epochs = FLAGS.num_epochs

    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    x = tf.placeholder("float", shape=[None, num_features])
    y_ = tf.placeholder("float", shape=[None, NUM_LABELS])
    
    # For the test data, hold the entire dataset in one constant node.
    test_data_node = tf.constant(test_data)

    # what kind of network do you want to build? One with hidden layers or not?
    hidden_layer = FLAGS.hidden_layer

    if hidden_layer:
        # Get the size of layer one. Either default to 1 or from command line
        num_hidden = FLAGS.num_hidden
        # Define and initialize the network.
        # Initialize the hidden weights and biases.
        w_hidden = init_weights([num_features, num_hidden], 'xavier', xavier_params=(num_features, num_hidden))
        b_hidden = init_weights([1,num_hidden],'zeros')
        # The hidden layer.
        hidden = tf.nn.tanh(tf.matmul(x,w_hidden) + b_hidden)
        # Initialize the output weights and biases.
        w_out = init_weights([num_hidden, NUM_LABELS], 'xavier', xavier_params=(num_hidden, NUM_LABELS))
        b_out = init_weights([1,NUM_LABELS],'zeros')
        # The output layer.
        y = tf.nn.softmax(tf.matmul(hidden, w_out) + b_out)
        

    else:

        # Define and initialize the network.
        # These are the weights that inform how much each feature contributes to
        # the classification.
        W = tf.Variable(tf.zeros([num_features,NUM_LABELS]))
        b = tf.Variable(tf.zeros([NUM_LABELS]))
        y = tf.nn.softmax(tf.matmul(x,W) + b)

    # Optimization.
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    # Evaluation.
    # match indices of largest term from tensors
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # Create a local session to run this computation.
    with tf.Session() as s:
        # Run all the initializers to prepare the trainable parameters.
        tf.initialize_all_variables().run()
        if verbose:
            print 'Initialized!'
            print
            print 'Training.'

        # Iterate and train.
        for step in xrange(num_epochs * train_size // BATCH_SIZE):
            if verbose:
                print step,
                
            #batch data processing logic
            offset = (step * BATCH_SIZE) % train_size
            batch_data = train_data[offset:(offset + BATCH_SIZE), :]
            batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
            #feeds since x and y_ are the placeholder inputs to all operations to compute trainstep
            train_step.run(feed_dict={x: batch_data, y_: batch_labels})
            if verbose and offset >= train_size-BATCH_SIZE:
                print

        # Give very detailed output.
        # if verbose:
        #     print
        #     print 'Weight matrix.'
        #     print s.run(W)
        #     print
        #     print 'Bias vector.'
        #     print s.run(b)
        #     print
        #     print "Applying model to first test instance."
        #     first = test_data[:1]
        #     print "Point =", first
        #     print "Wx+b = ", s.run(tf.matmul(first,W)+b)
        #     print "softmax(Wx+b) = ", s.run(tf.nn.softmax(tf.matmul(first,W)+b))
        #     print
            
        print "Accuracy:", accuracy.eval(feed_dict={x: test_data, y_: test_labels})

        test_out = []
        for index in range(len(test_data)):
            if test_labels[index][0] == 1.0:
                test_out.append(0)
            else:
                test_out.append(1)               
            print test_data[index, 0], test_data[index,1], test_out[index]
        
        plt.scatter(test_data[:, 0], test_data[:,1], s=40, c=test_out, cmap=plt.cm.Spectral)
        plt.show()


if __name__ == '__main__':
    tf.app.run()