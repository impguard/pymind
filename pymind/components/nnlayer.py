import numpy as np

class NNLayer(object):
  def __init__(self, num_input, activationfn):
    # Save input
    self.num_input = num_input
    self.activationfn = activationfn

  def activate(self, x):
    return self.activationfn.calc(x)
