import numpy as np
from nnetwork import NeuralNetwork
from nnlayer import NNLayer
from collections import deque
from scipy.optimize import fmin_l_bfgs_b

class NNTrainer(object):
  """ A NNTrainer takes a neural network and trains it given a training dataset.

  The NNTrainer can be initialized with a neural network. Then, depending on which training method
  is used, the NNTrainer will automatically update the neural network's weights to a optimal level.
  """
  def __init__(self, nn):
    self.nn = nn

  def createCostfn(self, X, y, learn_rate, errorfn):
    """ Creates a cost function that takes in a list of weights and returns the error cost.

    This method creates a method that, given a column vector of weights, first calculates the cost
    of using the list of weights with the provided dataset and then calculates the gradient of the
    cost using the list of weights with the provided dataset.

    The gradient of the cost if calculated using backpropogation, and the gradient utilizes the grad
    function in each of the neural network's layers as well as the grad function of the provided
    error function.

    Arguments:
    X -- The featureset with each column representing a feature vector for one training example.
    y -- The output vector with each column representing the output vector for one training example.
    Returns:
    A cost function that takes a list of weights.

    Note: The number of columns in X and y represent the number of training examples, so the number
    of columns of each must match up. Otherwise, a matrix dimension mismatch error will be raised.
    """
    def costfn(weights):
      """ The cost function created by createCostfn.

      Uses the parameters passed to createCostfn to calculate the cost and the gradient given a list
      of weights. The weights should be correctly shaped. The cost function depends on the error
      function provided and uses regularization. How much regularization is used can be tuned with
      the learning rate (lambda).
      """
      try:
        # Set the weights of the neural network
        self.nn.weights = weights

        # Helper variables
        m = float(X.shape(1)) # Number of training examples
        bias = 1 if self.nn.bias else 0

        # Part 1: Feed forward and get cost
        z, a = self.nn.feed_forward(X)
        h = a[-1] # Hypothesis
        err_vector = errorfn.calc(h, y)
        unreg_cost = (1. / m) * error.sum() # Unregularized cost
        sum_squared_weights = np.sum([np.square(weight[:, bias:]).sum() for weight in weights])
        reg_cost = (learn_rate / (2 * m)) * sum_squared_weights # Regularized cost
        cost = reg_cost + unreg_cost

        return cost
      except ValueError:
        print "Calculating cost of a neural network failed. Most likely due to dimension mismatch."
        raise

  def reshapeWeights(unrolled_weights):
    """ Reshapes an unrolled column vector of weights into the proper sizes.

    Assumes that the correct number of weights are passed, and uses the neural network's weight
    shapes and sizes to determine how to reshape the unrolled weights.
    Arguments:
    unrolled_weights -- A column vector of unrolled weights
    Returns:
    A list of matrices for each reshaped weight matrix

    Note: This method assumes that the size of unroll_weights is correct.
    """
    try:
      weights = list()
      curr_index = 0
      for weight in self.nn.weights:
        shape = weight.shape
        size = weight.size
        weight = np.matrix(unrolled_weights[curr_index:curr_index + size]).reshape(shape)
        weights.append(weight)
        curr_index += size
      return weights
    except ValueError:
      print "Reshaping weights failed. Most likely due to incorrect size of unrolled_weights."
      raise


  """
  def train(self, X, y):
    costFn = self.createCostFn(X, y)
    # Minimize!

    # x, f, d = fmin_l_bfgs_b(costFn, np.vstack(self.nn.weights).flatten())

  def createCostFn(self, X, y):
    def costFn(params):
      # Reshape params to get weights
      weights = list()
      curr_index = 0
      for weight in self.nn.weights:
        shape = weight.shape
        size = weight.size
        weights.append(np.matrix(params[curr_index:curr_index + size].reshape(shape)))
        curr_index += size

      # Set the weights of the neural network
      self.nn.weights = weights

      # Helper variables
      m = float(X.shape[1])
      bias = 1 if self.nn.bias else 0

      # Part 1: Feed-forward + Get error
      z, a = self.nn.feed_forward(X)
      # Errors
      error = self.err_fn.calc(a[-1], y)
      # Unregularized Cost
      cost = (1. / m) * error.sum()
      # Regularized
      cost += self.learn_rate / (2. * m) * sum(np.sum(np.power(weight[:, bias:], 2)) for weight in self.nn.weights)

      # Part 2: Backpropogation
      d = deque()

      lastD = np.multiply(self.nn.layers[-1].activationfn.grad(z[-1]), self.err_fn.grad(a[-1], y))

      d.appendleft(lastD)

      for i in range(len(self.nn.layers) - 2, 0, -1):
        fn = self.nn.layers[i].activationfn
        nextD = np.multiply(self.nn.weights[i][:, bias:].T * d[0], fn.grad(z[i]))
        d.appendleft(nextD)

      d.appendleft(None) # Filler so the indexes matchup

      # Get gradients
      grads = list()
      for i in range(len(self.nn.weights)):
        # Unregularized
        tmpGrad = (1. / m) * d[i+1] * a[i].T
        # Regularized
        tmpGrad[:, bias:] += (self.learn_rate / m) * tmpGrad[:, bias:]
        grads.append(tmpGrad)

      # Unroll gradients
      grad = np.hstack([g.flatten() for g in grads])

      return cost, grad.T
    return costFn
  """
