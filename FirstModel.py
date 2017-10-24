# To implement: Dropout, More conv layers

import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt

class FirstModel:

    def __init__(self, inp_w, inp_h, inp_d, keep_prob = 0.5):
        self._is_training = tf.placeholder(tf.bool)
        self._X = tf.placeholder(shape = [None, inp_w, inp_h, inp_d], dtype = tf.float32)
        # self._y = tf.placeholder(tf.int64, shape = [None])

        # First Convolutional Layer:
        self._inp_norm = tf.layers.batch_normalization(self._X, axis = 1, training = self._is_training)
        self._Wconv1 = tf.get_variable("Wconv1", shape = [7, 7, inp_d, 32], initializer = tf.contrib.layers.xavier_initializer())
        self._bconv1 = tf.get_variable("bconv1", shape = [32])
        self._z1 = tf.nn.conv2d(self._inp_norm, self._Wconv1, strides = [1, 1, 1, 1], padding = 'VALID') + self._bconv1
        self._h1 = tf.layers.batch_normalization(tf.nn.relu(self._z1), axis = 1, training = self._is_training)
        self._h1_max_pool = self.max_pool_2x2(self._h1)

        # Second convolutional layer:
        self._Wconv2 = tf.get_variable("Wconv2", shape = [5, 5, 32, 64], initializer = tf.contrib.layers.xavier_initializer())
        self._bconv2 = tf.get_variable("bconv2", shape = [64], initializer = tf.contrib.layers.xavier_initializer())
        self._z2 = tf.nn.conv2d(self._h1_max_pool, self._Wconv2, strides = [1, 1, 1 ,1], padding = 'VALID') + self._bconv2
        self._h2 = tf.layers.batch_normalization(tf.nn.relu(self._z2), axis = 1, training = self._is_training)
        self._h2_max_pool = self.max_pool_2x2(self._h2)
        self._h2_max_pool_flat = tf.reshape(self._h2_max_pool, [-1, 1600], 'h2_max_pool_flat')

        # First fully-connected layer:
        self._W1 = tf.get_variable("W1", shape = [1600, 1024], initializer = tf.contrib.layers.xavier_initializer())
        self._b1= tf.get_variable("b1", shape = [1024])
        self._fc1 = tf.matmul(self._h2_max_pool_flat, self._W1) + self._b1
        self._a2 = tf.nn.relu(self._fc1)
        self._fc1_normalized = tf.layers.batch_normalization(self._a2, axis = 1, training = self._is_training)

        # Dropout:
        # self._a2_dropout = tf.nn.dropout(self._a2, keep_prob)


        # Second fully-connected layer:
        self._W2 = tf.get_variable("W2", shape = [1024, 10], initializer = tf.contrib.layers.xavier_initializer())
        self._b2 = tf.get_variable("b2", shape = [10])
        self._a3 = tf.matmul(self._fc1_normalized, self._W2) + self._b2
        self._op = tf.layers.batch_normalization(self._a3, axis = 1, training = self._is_training)

    def ret_op(self):
        return self._op

    def run_model(self, session, predict, loss_val, Xd, yd,
                  epochs=1, batch_size=64, print_every=100,
                  training=None, plot_losses=False):
        # have tensorflow compute accuracy
        correct_prediction = tf.equal(tf.argmax(predict, 1), self._y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # shuffle indicies
        train_indicies = np.arange(Xd.shape[0])
        np.random.shuffle(train_indicies)

        training_now = training is not None

        # setting up variables we want to compute (and optimizing)
        # if we have a training function, add that to things we compute
        variables = [self._mean_loss, correct_prediction, accuracy]
        if training_now:
            variables[-1] = training

        # counter
        iter_cnt = 0
        for e in range(epochs):
            # keep track of losses and accuracy
            correct = 0
            losses = []
            # make sure we iterate over the dataset once
            for i in range(int(math.ceil(Xd.shape[0] / batch_size))):
                # generate indicies for the batch
                start_idx = (i * batch_size) % Xd.shape[0]
                idx = train_indicies[start_idx:start_idx + batch_size]

                # create a feed dictionary for this batch
                feed_dict = {self._X: Xd[idx, :],
                             self._y: yd[idx],
                             self._is_training: training_now}
                # get batch size
                actual_batch_size = yd[idx].shape[0]

                # have tensorflow compute loss and correct predictions
                # and (if given) perform a training step
                loss, corr, _ = session.run(variables, feed_dict=feed_dict)

                # aggregate performance stats
                losses.append(loss * actual_batch_size)
                correct += np.sum(corr)

                # print every now and then
                if training_now and (iter_cnt % print_every) == 0:
                    print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}" \
                          .format(iter_cnt, loss, np.sum(corr) / actual_batch_size))
                iter_cnt += 1
            total_correct = correct / Xd.shape[0]
            total_loss = np.sum(losses) / Xd.shape[0]
            print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}" \
                  .format(total_loss, total_correct, e + 1))
            if plot_losses:
                plt.plot(losses)
                plt.grid(True)
                plt.title('Epoch {} Loss'.format(e + 1))
                plt.xlabel('minibatch number')
                plt.ylabel('minibatch loss')
                plt.show()
        return total_loss, total_correct


    # Define a max pool layer with size 2x2, stride of 2 and same padding.
    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

    # Predict:
    def predict(self, X):
        with tf.Session() as sess:
            with tf.device("/cpu:0"):
                tf.global_variables_initializer().run()
                ans = sess.run(self._op, feed_dict = {self._X : X, self._is_training : True})
                return ans

    # Train:
    def train(self, X, y):
        self._y = tf.placeholder(tf.int64, shape = [None])
        self._mean_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self._op, labels = tf.one_hot(self._y, 10)))
        self._optimizer = tf.train.AdamOptimizer(1e-4)
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            self._train_step = self._optimizer.minimize(self._mean_loss)
        self._sess = tf.Session()

        self._sess.run(tf.global_variables_initializer())
        print('Training')
        self.run_model(self._sess, self._op, self._mean_loss, X, y, 1, 64, 100, self._train_step)

    def evaluate (self, X, y):
        self.run_model(self._sess, self._op, self._mean_loss, X, y, 1, 64)

