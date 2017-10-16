from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange
import random

# We implement the stable version of softmax:
def softmax(X):
    ret = []
    for x in X:
        e_x = np.exp(x - np.max(x))
        ret.append(e_x/e_x.sum())
    return np.array(ret)

def relu_grad(X):
    grd = []
    for x in X:
        row = []
        for m in x:
            if m > 0:
                row.append(1)
            else:
                row.append(0)
        grd.append(row)
    return np.array(grd)

# Simple implementation of a one-hot encoder:
def one_hot_encoder(y, output_size):
    ret_mat = []
    for i in y:
        ret_row = np.zeros(output_size)
        ret_row[i] = 1
        ret_mat.append(ret_row)
    return np.array(ret_mat)

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    fc1 = np.add(np.matmul(X, W1), b1) # shape: (N, H)
    hidden1 = np.maximum(fc1, np.zeros(shape = fc1.shape)) # shape: (N,H)
    scores = np.add(np.matmul(hidden1, W2), b2) # shape: (N, C)
    activation2 = softmax(scores) # shape: (N, C)
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss.                                                          #
    #############################################################################
    y = one_hot_encoder(y, b2.shape)
    print(activation2)
    # loss = np.mean()
    reg_term = reg * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    loss = np.mean(np.sum((y * np.log(activation2)), axis = 1) * (-1)) +  reg_term
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    # This is from the tutorial on Stanford website:
    # f = lambda W : self.loss(X, y, reg = reg)[0]
    err_vec = (activation2 - y) / N # shape (N,C)
    err_W2 =  np.matmul(hidden1.T, err_vec) + W2 * reg * 2 # shape (H, C)
    err_b2 = np.sum(err_vec, axis = 0) # shape (C,)
    err_vec_2 =  np.matmul(err_vec, W2.T) # shape (N, H)
    err_vec_2 = err_vec_2 * relu_grad(fc1)
    err_W1 = np.matmul(X.T, err_vec_2) + W1 * reg * 2 # shape (D, H)
    err_b1 = np.sum(err_vec_2, axis = 0) # shape (H,)
    grads['W1'] = err_W1
    grads['W2'] = err_W2
    grads['b1'] = err_b1
    grads['b2'] = err_b2
    # for key in self.params:
    #     h = 0.00001
    #     # grad[key] = eval_num_grad(f, )
    #     x = self.params[key] # The variable that we are taking gradient with respect to.
    #     grad = np.zeros(x.shape)
    #     it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    #     while not it.finished:
    #         ix = it.multi_index
    #         old_value = x[ix]
    #         x[ix] = old_value + h
    #         sc1 = self.loss(X)
    #         ac1 = softmax(sc1)
    #         loss1 = np.mean(np.sum((y * np.log(ac1)), axis = 1) * (-1)) +  reg * (np.sum(np.square(self.params['W1'])) + np.sum(np.square(self.params['W2'])))
    #         # x[ix] = old_value - h
    #         # sc2 = self.loss(X)
    #         # ac2 = softmax(sc2)
    #         # loss2 = np.mean(np.sum((y * np.log(ac2)), axis = 1) * (-1)) +  reg * (np.sum(np.square(self.params['W1'])) + np.sum(np.square(self.params['W2'])))
    #         x[ix] = old_value
    #
    #         # approx. the partial derivative
    #         grad[ix] = (loss1 - loss) / h
    #         it.iternext()
    #     grads[key] = grad
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in xrange(num_iters):
      X_batch = []
      y_batch = []

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      indices = random.sample(range(num_train), batch_size)
      for i in indices:
          X_batch.append(X[i])
          y_batch.append(y[i])
      X_batch = np.array(X_batch)
      y_batch = np.array(y_batch)
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      for key in self.params:
          self.params[key] = self.params[key] - learning_rate * grads[key]
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    fc1 = np.matmul(X, self.params['W1']) + self.params['b1']
    hidden1 = np.maximum(fc1, np.zeros(shape = fc1.shape)) # shape: (N,H)
    scores = np.add(np.matmul(hidden1, self.params['W2']), self.params['b2']) # shape: (N, C)
    activation2 = softmax(scores) # shape: (N, C)
    y_pred = np.argmax(scores, axis = 1)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred
