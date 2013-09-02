import numpy as np
from ..util import initRandParams

class NNetworkLayer(object):
  def __init__(self, num_input, activation_fn):
    # Save input
    self.num_input = num_input
    self.activation_fn = activation_fn

  def activate(self, x):
    if x.shape[0] != self.num_input:
      raise Exception("The input x has the wrong number of rows. Expected %s." % self.num_input)
    return self.activation_fn.calc(x)
