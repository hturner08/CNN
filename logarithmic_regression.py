# Logarithmic Regression

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
from random import *
import math

LOGDIR = "./tmp/logarithmic_regression"

# Parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1


# have x be a vertical matrix with all the data points, with 1 at front
# have w be a horiz matrix with all data points, with b at front
x = tf.placeholder(tf.float32, shape=[None, 1])
y = tf.placeholder(tf.float32, shape = [None, 1])
arrw = []
for i in range(60):
    arrw.append(random())

# it won't take numpy arrays, it won't take lists
W = tf.get_variable(name = 'W', dtype = tf.float32,shape = [1, 1], initializer=tf.contrib.layers.xavier_initializer())
b = tf.get_variable(name = 'b', dtype = tf.float32, initializer = tf.zeros([1]))


# logarithmic model prediction
pred = tf.nn.sigmoid(x * W + b)

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# define sample training data
x_train = np.array([[  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,
         11.,  12.,  13.,  14.,  15.,  16.,  17.,  18.,  19.,  20.,  21.,
         22.,  23.,  24.,  25.,  26.,  27.,  28.,  29.]]).T.reshape(30,1)

y_train = np.array([[  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   1.,   0.,   0.,   1.,   0.,   1.,   1.,   1.,   1.,   1.,
         1.,   1.,   1.,   1.,   1.,   1.,   1.,   1.]]).T.reshape(30,1)
def makeSample(size):
    for x in range(training):

with tf.Session() as sess:
    # Initialize the variables (i.e. assign their default value)
    tf.global_variables_initializer().run()

    # Begin training
    for _ in range(100):
        sess.run(optimizer, {x: x_train, y : y_train})

    curr_W, curr_b, curr_loss = sess.run([W, b, cost], {x: x_train, y: y_train})
    print("W: %s b: %s cost: %s"%(curr_W, curr_b, curr_loss))

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    print("Prediction:", tf.argmax(pred,1),"Label" tf.argmax(y,1))
    # Calculate accuracy for 3000 examples
    # BUG: accuracy always 1?
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: x_train, y: y_train}))

    writer = tf.summary.FileWriter(LOGDIR, sess.graph)
    writer.add_graph(sess.graph)
