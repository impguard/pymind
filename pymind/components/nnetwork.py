import numpy as np
from nnlayer import NetworkLayer
from ..util import *
from collections import deque

class NeuralNetwork:

  # Create a neural network by passing in a list of options.
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
  def __init__(self, **params):
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
    self.layers = deque()
    fn_count = range(len(self.activation_fn)).__iter__()

    # Input layer
    self.addLayer(self.input_units, fn_count.next())

    # Hidden layers
    for units in self.hidden_units:
      self.addLayer(units, fn_count.next())

    # Output layer
    self.addLayer(self.output_units, fn_count.next())

    # ----- Create network weights ----- #

  def addLayer(units, activation_fn_index):
    new_layer = NetworkLayer(units, self.activation_fn[activation_fn_index])
    self.layers.appendleft(new_layer)

