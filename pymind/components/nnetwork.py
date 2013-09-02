import numpy as np
from nnlayer import NNetworkLayer
from ..util import initRandParams

class NeuralNetwork(object):

  # Create a neural network by passing in a dictionary of parameters.
  #
  # The parameters that are passed in include:
  #   input_units (number): number of input units
  #   output_units (number): number of output units
  #   hidden_units (number|list:number): number of hidden layer units; a list represents the number of hidden layer
  #     units per hidden layer (which will be inferred from the number of elements in the list)
  #   activation_fn (list:function): a list of activation functions for each layer in the network; the number of
  #     functions should equal the number layers inferred from the previous parameters
  #   bias (bool): whether a bias unit should be introduced in each layer
  #
  # Note: Only the hidden_units parameter is optional
  # Note: The activation functions must be
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
    self.activation_fn = params['activation_fn']

    # ----- Create layers ----- #
    self.layers = list()
    fn_count = range(len(self.activation_fn)).__iter__()

    # Input layer
    self.addLayer(self.input_units, fn_count.next())

    # Hidden layers
    for units in self.hidden_units:
      self.addLayer(units, fn_count.next())

    # Output layer
    self.addLayer(self.output_units, fn_count.next())

    # ----- Create network weights ----- #
    self.weights = list()
    bias = 1 if self.bias else 0
    for i in range(len(self.layers)):
      if i == len(self.layers) - 1:
        break;
      num_input = self.layers[i].num_input + bias
      num_output = self.layers[i+1].num_input
      self.weights.append(initRandParams(num_input, num_output))

  # Helper method for initialization
  def addLayer(self, units, activation_fn_index):
    new_layer = NNetworkLayer(units, self.activation_fn[activation_fn_index])
    self.layers.append(new_layer)

  # For testing purposes only
  def setWeight(self, index, weight):
     self.weights[index] = weight

  # Useful methods for using a NeuralNetwork
  def getWeights(self, index):
    return self.weights[index]

  def getLayer(self, index):
    return self.layers[index]

  def feed_forward(self, x):
    if x.shape[0] != self.input_units:
      raise Exception("The input x has the wrong number of rows. Expected %s." \
        % self.input_units)

    curr_z = x
    curr_a = None
    z = list()
    a = list()
    weights = self.weights.__iter__()

    for i in range(len(self.layers)):
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
