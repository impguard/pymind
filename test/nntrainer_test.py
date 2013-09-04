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
  cgrdvec,grdvec = cgrd.T,grd.T
  for i in range(len(cgrdvec)):
    np.testing.assert_array_almost_equal(cgrdvec[i], grdvec[i], decimal = 3,
      err_msg = "The output grd at index %d should be \n %r \n != %r" % (i, cgrdvec[i], grdvec[i]))

"""
Computes the gradient of the weight vector _weights_, given a cost
function _costFn_ and an optional epsilon _e_. Smaller epsilon values
allow more accurate estimations of the gradient.
"""
def computeNumericalGradient(costFn, weights, e=0.01):
  wsize = len(weights)
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
  return np.matrix(grd).reshape((1,wsize))
