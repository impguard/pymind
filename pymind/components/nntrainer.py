import numpy as np
from nnetwork import NeuralNetwork
from nnlayer import NNLayer
from collections import deque
from scipy.optimize import fmin_l_bfgs_b

class NNTrainer(object):
  def __init__(self, nn, err_fn, learn_rate):
    self.nn = nn
    self.err_fn = err_fn
    self.learn_rate = learn_rate

  def train(self, X, y):
    costFn = self.createCostFn(X, y)
    # Minimize!
    x, f, d = fmin_l_bfgs_b(costFn, np.vstack(self.nn.weights).flatten())

  """ Creates a cost function for a given dataset (X, y)

  Args:
      X: The featureset with columns representing a feature vector and rows representing training examples
      y: The output vector with columns representing the output vector and rows representing training examples
  Returns:
      a cost function that takes an unrolled set of params
  Note:
      the returned cost function returns the tuple (cost, gradient) based on the error function used for the trainer
  """
  def createCostFn(self, X, y):
    def costFn(params):
      # Flip since scipy passes in a row vector
      params = params.T

      # Reshape params to get weights
      weights = list()
      curr_index = 0
      for weight in self.nn.weights:
        shape = weight.shape
        size = weight.size
        weights.append(np.matrix(params[curr_index, curr_index + size].reshape(shape)))
        curr_index += size

      # Set the weights of the neural network
      self.nn.weights = weights

      # Helper variables
      m = X.shape[1]
      bias = 1 if self.bias else 0

      # Part 1: Feed-forward + Get error
      z, a = self.nn.feed_forward(X)
      # Errors
      error = self.err_fn.calc(a[-1], y)
      # Unregularized Cost
      cost = (1. / m) * sum(error)
      # Regularized
      cost += (self.learn_rate / 2. * m) * sum(sum(np.power(weight[:, bias:], 2)) for weight in self.nn.weights)

      # Part 2: Backpropogation
      d = deque()
      lastD = np.multiply(self.nn.layers[-1].activationfn.grad(z[-1]), self.err_fn.grad(a[-1], y))
      d.appendleft(lastD)

      for i in range(len(self.nn.layers) - 2, 1, -1):
        fn = self.nn.layers[i].activationfn
        nextD = np.multiply(self.nn.weights[i][:, bias:].T * d[0], fn.grad(z[i]))
        d.appendleft(nextD)

      d.appendleft(None) # Filler so the indexes matchup

      # Get gradients
      grads = list()
      for i in range(len(self.nn.weights)):
        # Unregularized
        tmpGrad = (1. / m) * d[i+1] * a[i]
        # Regularized
        tmpGrad[:, bias:] += (self.learn_rate / m) * tmpGrad[:, bias:]
        grads.append(tmpGrad)

      # Unroll gradients
      grad = np.vstack(grads).flatten()

      return cost, grad
    return costFn
