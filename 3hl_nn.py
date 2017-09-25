import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
LOGDIR = "./tmp/3hl_nn"


# can modify node numbers
n_nodes_hl1 = 50
n_nodes_hl2 = 50
n_nodes_hl3 = 50

n_classes = 10
batch_size = 500

# input data, height by width
x = tf.placeholder(name = 'x', dtype = tf.float32, shape = [None,784]) # x = [:, 784]


def neural_net_model(data):
    # inputdata * weights + biases

    hl1 = {'weights': tf.get_variable(name = 'w1', dtype=tf.float32, initializer=tf.random_normal([784,n_nodes_hl1])),
            'biases': tf.get_variable(name = 'b1', dtype = tf.float32, initializer= tf.random_normal([n_nodes_hl1]))}

    hl2 = {'weights': tf.get_variable(name = 'w2', dtype=tf.float32, initializer=tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
            'biases':  tf.get_variable(name = 'b2', dtype = tf.float32, initializer= tf.random_normal([n_nodes_hl2]))}

    hl3 = {'weights': tf.get_variable(name = 'w3', dtype=tf.float32, initializer=tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
            'biases':  tf.get_variable(name = 'b3', dtype = tf.float32, initializer= tf.random_normal([n_nodes_hl3]))}

    outputlayer = {'weights': tf.get_variable(name = 'w4', dtype=tf.float32, initializer=tf.random_normal([n_nodes_hl3,n_classes])),
            'biases': tf.get_variable(name = 'b4', dtype = tf.float32, initializer=tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data, hl1['weights']), hl1['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hl2['weights']), hl2['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hl3['weights']), hl3['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3, outputlayer['weights']), outputlayer['biases'])
    return output

def run_neural_net(x):
    prediction = neural_net_model(x)
    y = tf.placeholder(name = 'y', dtype=tf.float32, shape = [i for i in prediction.shape])

    cost = tf.reduce_mean(tf.square(y - prediction))
    optimizer = tf.train.AdamOptimizer(0.1).minimize(cost)

    num_epochs = 5 # the larger the slower

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(num_epochs):
            total_correct = 0;
            for i in range(int(mnist.train.num_examples/batch_size)):
                # Training set
                epochx, epochy = mnist.train.next_batch(batch_size)
                sess.run(optimizer, feed_dict = {x: epochx, y: epochy})
                # Testing set
                epochx, epochy = mnist.train.next_batch(batch_size)
                y_prob = sess.run(y, feed_dict = {x: epochx, y: epochy})
                correct_pred = 0
                for i in range(batch_size):
                     if (np.absolute(epochy[i,0] - y_prob[i,0]) < .01):
                        correct_pred += 1
                total_correct += correct_pred

            print("Epoch " + str(epoch) + "'s accuracy: " + str(total_correct/mnist.train.num_examples))
            
        writer = tf.summary.FileWriter(LOGDIR, sess.graph)
        writer.add_graph(sess.graph)

run_neural_net(x)
