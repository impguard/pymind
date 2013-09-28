import numpy as np
from nnlayer import NNLayer

class NeuralNetwork(object):
  """ Create a neural network by passing in a dictionary of parameters.

  The parameters that are passed in include:
    input_units (number): number of input units
    output_units (number): number of output units
    hidden_units (number|list:number): number of hidden layer units; a list represents the number of hidden layer
      units per hidden layer (which will be inferred from the number of elements in the list)
    activationfn (list:function): a list of activation functions for each layer in the network; the number of
      functions should equal the number layers inferred from the previous parameters
    bias (bool): whether a bias unit should be introduced in each layer

  Note: Only the hidden_units parameter is optional
  """
  def __init__(self, params):
    # ----- Initialize parameters ------ #
    if "hidden_units" not in params:
      self.hidden_units = []
    elif type(params['hidden_units']) is int:
      # Convert integer to one element list
      self.hidden_units = [params['hidden_units']]
    else:
      self.hidden_units = params['hidden_units']
    # Save other inputs
    self.input_units = params['input_units']
    self.output_units = params['output_units']
    self.bias = params['bias']
    self.activationfn = params['activationfn']

    # ----- Create layers ----- #
    self.layers = list()
    fn_count = xrange(len(self.activationfn)).__iter__()

    # Input layer
    self.addLayer(self.input_units, fn_count.next())

    # Hidden layers
    for units in self.hidden_units:
      self.addLayer(units, fn_count.next())

    # Output layer
    self.addLayer(self.output_units, fn_count.next())

    # ----- Create network weights ----- #
    self.resetWeights()

  # Helper methods for initialization
  def addLayer(self, units, activationfn_index):
    new_layer = NNLayer(units, self.activationfn[activationfn_index])
    self.layers.append(new_layer)

  def initRandWeights(self, num_input, num_output):
    eps_init = np.sqrt(6) / np.sqrt(num_input + num_output)
    return np.matrix(np.random.rand(num_output, num_input) * 2 * eps_init - eps_init)

  # Front-end methods for using a NeuralNetwork
  def resetWeights(self):
    self.weights = list()
    bias = 1 if self.bias else 0
    for i in xrange(len(self.layers) - 1):
      num_input = self.layers[i].num_input + bias
      num_output = self.layers[i+1].num_input
      self.weights.append(self.initRandWeights(num_input, num_output))

  def feedForward(self, x):
    """ Runs the feed forward process with this neural network.

    This function takes in the input vector x and runs the feed forward process. However, x must be
    the correct dimension matching the number of input units this neural network accepts.

    Note: The actual output of the process is stored in a[-1]

    Arguments:
    x -- A column vector with dimensions (self.input_units x m)
    Returns:
    z -- A list of column vectors representing the inputs to each layer in the neural network
    a -- A list of column vectors representing the outputs of each layer in the neural network
    """
    try:
      curr_z = x
      curr_a = None
      z = list()
      a = list()
      weights = self.weights.__iter__()

      for i in xrange(len(self.layers)):
        layer = self.layers[i]
        isOutput = i == len(self.layers) - 1
        # Activation Step
        curr_a = layer.activate(curr_z)
        if self.bias and not isOutput:
          ones = np.ones((1, curr_a.shape[1]))
          curr_a = np.vstack((ones, curr_a))
        z.append(curr_z)
        a.append(curr_a)
        if isOutput:
          break
        # Feed Forward Step
        curr_z = weights.next() * curr_a

      return z, a
    except ValueError:
      print "Feed forward process failed. Most likely due to matrix dimension mismatch."
      raise

  def activate(self, x):
    """ Runs the feed forward process with this neural network, but just returns the output.

    This function takes in the input vector x and runs the feed forward process. However, it will
    only return the output of the neural network. Use feedForward if the outputs of every layer in
    the neural network in desired.

    Arguments:
    x -- A column vector with dimensions (self.input_units x m)
    Returns:
    The output vector as a column vector with dimensions (self.output_units x m)
    """

    z, a = self.feedForward(x)
    return a[-1]

  def calculateCost(self, x, y, errfn, learn_rate):
    """ Internal helper method for quickly calculating the cost of a neural network.

    Will perform the feed forward process and compare the output to a expected output, using an
    errfn to determine the actual cost.

    Note: Use the cost method directly on the neural network to simply get the cost.

    Arguments:
    x -- A column vector with dimensions (self.input_units x m) representing the input
    y -- A column vector with dimensions (self.output_units x m) representing the expected output
    errfn -- An errfn
    learn_rate -- The learning rate used in regularization
    Returns:
    z -- A list of column vectors representing the inputs to each layer in the neural network
    a -- A list of column vectors representing the outputs to each layer in the neural network
    cost -- The resulting cost of the neural network when compared to the expected output
    """
    # Helper variables
    m = float(x.shape[1]) # Number of training examples
    bias = 1 if self.bias else 0

    z, a = self.feedForward(x)
    h = a[-1] # Hypothesis
    err_vector = errfn.calc(h, y)
    unreg_cost = (1. / m) * err_vector.sum() # Unregularized cost
    sum_squared_weights = np.sum([np.square(weight[:, bias:]).sum() for weight in self.weights])
    reg_cost = (learn_rate / (2 * m)) * sum_squared_weights # Regularized cost
    cost = reg_cost + unreg_cost

    return z, a, cost
