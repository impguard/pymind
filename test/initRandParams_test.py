import numpy as np
from pymind.util import initRandParams

def testInitRandParams():
  params = initRandParams(3, 3)
  assert params.shape == (3, 3), \
    "The shape of the parameter matrix generated for 3 inputs and 3 outputs should be (3, 3): %s" % params.shape
