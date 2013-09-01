import numpy as np
from ..util import initRandParams

class NetworkLayer:
  def __init__(self, num_input, activation_fnc):
    # Save input
    self.num_input = num_input
    self.activation_fnc = np.frompyfunc(activation_fnc, 1, 1)

  def activate(self, input):
    if input.shape != (self.num_input, 1):
      raise DimensionError("The input has the wrong shape. Expected %s." % (num_input, 1), (num_input, 1))
    return self.activation_fnc(input)
