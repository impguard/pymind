import numpy as np
from numpy.random import random as nprandom
from pymind.components import NeuralNetwork, NNTrainer
from pymind.activationfn import *
from pymind.errfn import *

def testGradient():
  # Create Neural NeuralNetwork
  params = {
    "input_units": 10,
    "output_units": 6,
    "hidden_units": 8,
    "activationfn": [identity, sigmoid, sigmoid],
    "bias": True
  }
  nnet = NeuralNetwork(params)
  # Create NNTrainer
  trainer = NNTrainer(nnet,logitError,1)
  # Create Random Data
  X = nprandom((10,1000))
  y = nprandom((6,1000))
  # Create Cost Function
  costFn = trainer.createCostFn(X,y)
  # Run Cost Function and retrieve gradient
  wvec = nprandom(sum([w.size for w in nnet.weights]))
  cst,grd = costFn(wvec)
  # Run computeNumericalGradient with the network costs and weights
  cgrd = computeNumericalGradient(costFn,wvec)
  for i in range(len(cgrd)):
    np.testing.assert_array_almost_equal(cgrd[i], grd[i], decimal = 4,
      err_msg = "The output grd at index %d should be \n %r \n != %r" % (i, cgrd[i], grd[i]))

"""
Computes the gradient of the weight vector _weights_, given a cost
function _costFn_ and an optional epsilon _e_. Smaller epsilon values
allow more accurate estimations of the gradient.
"""
def computeNumericalGradient(costFn, weights, e=0.01):
  grd = np.empty(len(weights))
  for i in xrange(len(weights)):
    w = weights[i]
    w_inc = w + e
    w_dec = w - e
    weights[i] = w_inc
    c_inc,_ = costFn(weights)
    weights[i] = w_dec
    c_dec,_ = costFn(weights)
    grd[i] = (c_inc - c_dec)/(2*e)
    weights[i] = w
  return grd
