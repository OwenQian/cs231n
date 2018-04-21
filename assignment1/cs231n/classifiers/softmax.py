import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]

  scores = X.dot(W)
  # normalization trick to avoid numeric stability issues
  scores -= np.max(scores, axis=1).reshape(num_train, 1)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_train):
    score_sum = np.sum(np.exp(scores[i]))
    # softmax of all the classes
    softmax = np.exp(scores[i]) / score_sum

    # loss is calculated using only the score of the correct class
    L_i = -np.log(softmax[y[i]])
    loss += L_i

    for j in range(num_classes):
        dW[:,j] += X[i] * softmax[j]
        if j == y[i]:
            dW[:, j] -= X[i]
      
  loss /= num_train
  loss += reg * np.sum(W*W)

  dW /= num_train
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_train = X.shape[0]
  num_classes = W.shape[1]
  correct_class_indices = (np.arange(num_train), y)

  scores = X.dot(W)
  normalized_scores = scores - np.max(scores, axis=1).reshape(num_train, 1)

  scores_sum = np.sum(np.exp(normalized_scores), axis=1)

  softmax = np.divide(np.exp(normalized_scores), scores_sum.reshape(num_train, 1))

  loss = np.sum(-np.log(softmax[correct_class_indices])) / num_train + reg * np.sum(W*W)

  # For the correct class we have to subtract an extra X term
  softmax[correct_class_indices] -= 1

  dW = (X.T).dot(softmax) / num_train + 2*reg*W

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

