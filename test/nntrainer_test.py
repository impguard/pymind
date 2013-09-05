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
  X = np.matrix(np.exp(-np.arange(-1,1,2.0/5000).reshape((10,500))**2))
  y = np.matrix(np.exp(-np.arange(-1,1,2.0/3000).reshape((6,500))**2))
  # Create Cost Function
  costFn = trainer.createCostFn(X,y)
  # Run Cost Function and retrieve gradient
  wsize = sum([w.size for w in nnet.weights])
  wvec = np.sin(3*np.arange(-1,1,2.0/wsize))
  cst,grd = costFn(wvec)
  # Run computeNumericalGradient with the network costs and weights
  cgrd = computeNumericalGradient(costFn,wvec)
  for i in range(len(cgrd)):
    np.testing.assert_array_almost_equal(cgrd[i], grd[i], decimal = 3,
      err_msg = "The output grd at index %d should be \n %r \n != %r" % (i, cgrd[i], grd[i]))

"""
Computes the gradient of the weight vector _weights_, given a cost
function _costFn_ and an optional epsilon _e_. Smaller epsilon values
allow more accurate estimations of the gradient.
"""
def computeNumericalGradient(costFn, weights, e=0.001):
  wsize = weights.size
  grd = np.empty(wsize)
  for i in xrange(wsize):
    w = weights[i]
    w_inc = w + e
    w_dec = w - e
    weights[i] = w_inc
    c_inc,_ = costFn(weights)
    weights[i] = w_dec
    c_dec,_ = costFn(weights)
    grd[i] = (c_inc - c_dec)/(2*e)
    weights[i] = w
  return np.matrix(grd).reshape((wsize,1))
