import numpy as np
from pymind.components.nnlayer import *

def testNetworkLayer():
  # Create a network layer with the identity function
  layer = NetworkLayer(3, lambda x: x)
  z = np.matrix([3, 3, 3]).T
  a = layer.activate(z)
  np.testing.assert_array_equal(z, a,
    err_msg = "A network layer with an identity activation function should return an output identical with the input")
