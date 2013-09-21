import numpy as np
from nnetwork import NeuralNetwork
from nnlayer import NNLayer
from collections import deque
from scipy.optimize import minimize

class NNTrainer(object):
  """ A NNTrainer takes a neural network and trains it given a training dataset.

  The NNTrainer can be initialized with a neural network. Then, depending on which training method
  is used, the NNTrainer will automatically update the neural network's weights to a optimal level.
  """
  def __init__(self, nn):
    self.nn = nn

  def train(self, X, y, learn_rate, errorfn, iterations = 50, disp = False):
    costfn = self.createCostfn(X, y, learn_rate, errorfn)
    def flattenedCostfn(weights):
      """ Wrapper function that flattens inputs to pass to a scipy optimization function

      Arguments:
      weights -- a row vector of weights
      Returns:
      The cost and the gradient of the neural network given the weights.
      """
      cost, grad = costfn(self.reshapeWeights(np.matrix(weights)))
      return cost, np.array(self.unrollWeights(grad).T)[0]

    unrolled_weights_array = np.array(self.unrollWeights(self.nn.weights).T)[0]

    result = minimize(flattenedCostfn, unrolled_weights_array,
      method='BFGS',
      options = {'maxiter' :iterations, 'disp':disp},
      jac =True)

    if result.success:
      min_weights = self.reshapeWeights(np.matrix(result.x).T)
      self.nn.weights = min_weights
      return min_weights
    else:
      return None

  def createCostfn(self, X, y, learn_rate, errorfn, computeGrad=True):
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
    learn_rate -- The learning rate for regularization
    errorfn -- Error function used when computing cost
    computeGrad -- Whether or not to compute the gradient or not in addition to the cost
    Returns:
    A cost function that takes a list of weights.

    Note: The number of columns in X and y represent the number of training examples, so the number
    of columns of each must match up. Otherwise, a matrix dimension mismatch error will be raised.
    """
    learn_rate = float(learn_rate)

    def costfn(weights):
      """ The cost function created by createCostfn.

      Uses the weights passed to createCostfn to calculate the cost and the gradient given a list
      of weights. The weights should be correctly shaped. The cost function depends on the error
      function provided and uses regularization. How much regularization is used can be tuned with
      the learning rate (lambda).

      Returns:
      The cost and, if computeGrad is True, the gradient of the neural network.
      """
      try:
        # Set the weights of the neural network
        self.nn.weights = weights

        # Helper variables
        m = float(X.shape[1]) # Number of training examples
        bias = 1 if self.nn.bias else 0

        # Part 1: Feed forward and get cost
        z, a = self.nn.feed_forward(X)
        h = a[-1] # Hypothesis
        err_vector = errorfn.calc(h, y)
        unreg_cost = (1. / m) * err_vector.sum() # Unregularized cost
        sum_squared_weights = np.sum([np.square(weight[:, bias:]).sum() for weight in weights])
        reg_cost = (learn_rate / (2 * m)) * sum_squared_weights # Regularized cost
        cost = reg_cost + unreg_cost

        if computeGrad:
          # Part 2: Backpropogation and get grad
          d = deque()

          # Calculate deltas
          d.appendleft(np.multiply(self.nn.layers[-1].activationfn.grad(z[-1]), errorfn.grad(h, y)))

          for i in range(len(self.nn.layers) - 2, 0, -1):
            prevD = d[0]
            activationfn = self.nn.layers[i].activationfn
            weight = weights[i]
            d.appendleft(np.multiply(activationfn.grad(z[i]), weight[:, bias:].T * prevD))

          d.appendleft(None) # Filler since d0 is unimportant

          # Calculate gradients
          grads = list()
          for i in range(len(self.nn.weights)):
            # Unregularized gradient
            unreg_grad = (1. / m) * d[i+1] * a[i].T
            # Regularized gradient excluding bias
            reg_grad = (learn_rate / m) * weights[i][:, bias:]
            # Create gradient
            grad = unreg_grad
            grad[:, bias:] += reg_grad
            grads.append(grad)

          return cost, grads
        else:
          return cost
      except ValueError:
        print "Calculating cost of a neural network failed. Most likely due to dimension mismatch."
        raise
    return costfn

  def reshapeWeights(self, unrolled_weights):
    """ Reshapes an unrolled column or row vector of weights into the proper sizes.

    Assumes that the correct number of weights are passed, and uses the neural network's weight
    shapes and sizes to determine how to reshape the unrolled weights.

    Arguments:
    unrolled_weights -- A column or row vector of unrolled weights
    Returns:
    A list of matrices for each reshaped weight matrix.

    Note: This method assumes that the size of unrolled_weights is correct.
    """
    try:
      unrolled_weights = unrolled_weights if unrolled_weights.shape[1] == 1 else unrolled_weights.T
      weights = list()
      curr_index = 0
      for weight in self.nn.weights:
        shape = weight.shape
        size = weight.size
        unrolled = np.matrix(unrolled_weights[curr_index:curr_index + size])
        weight = unrolled.reshape(shape)
        weights.append(weight)
        curr_index += size
      return weights
    except ValueError:
      print "Reshaping weights failed. Most likely due to incorrect size of unrolled_weights."
      raise

  def unrollWeights(self, weights):
    """ Unrolls a list of weights into a column vector.

    Presumes that a list of weights of the proper dimensions are passed. Does not copy the passed
    weights.

    Returns:
    A column vector representing the unrolled weight vector.
    """
    unrolled_list = [weight.ravel() for weight in weights]
    return np.hstack(unrolled_list).T
