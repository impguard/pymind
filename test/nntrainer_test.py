import numpy as np
from numpy.random import random as nprandom
from pymind.components import NeuralNetwork, NNTrainer
from pymind.activationfn import *
from pymind.errfn import *

def generateMatrix(rows, columns, fn):
  """ Generates a matrix of a given shape using the passed function.

  Uses numpy.arange to create a row vector from 0 to the desired size, then reshapes the matrix to
  the desired dimensions. Finally, applies the supplied numpy universal function to the matrix,
  which by default is the sine function.

  Arguments:
  rows -- the number of desired rows
  columns -- the number of desired columns
  fn -- a numpy universal function (default: numpy.sin)
  Returns
  A matrix with dimensions (rows x columns) mutated with fn.
  """
  size = rows * columns
  matrix = np.arange(size).reshape(rows, columns)
  return fn(matrix)

def testCost1():
  """ Tests cost function of the nntrainer class. """

  # Test co
  params = {
    "input_units": 10,
    "output_units": 6,
    "hidden_units": 8,
    "activationfn": [identity, sigmoid, sigmoid],
    "bias": True
  }
  nnet = NeuralNetwork(params)
  trainer = NNTrainer(nnet)

  # Create cost function parameters
  X = generateMatrix(10, 3)
  y = generateMatrix(6, 3)
  learn_rate = 0
  errfn = logitError

  # Create cost function
  costfn = trainer.createCostfn(X, y, learn_rate, errfn)

  # Generate input weights
  weights = list()
  weights.append(generateMatrix(8, 11))
  weights.append(generateMatrix(6, 9))

  # Run cost function
  pass

"""
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
  X = np.matrix(np.sin(-np.arange(-1,1,2.0/5000).reshape((10,500))**2))
  y = np.matrix(np.sin(-np.arange(-1,1,2.0/3000).reshape((6,500))**2))
  # Create Cost Function
  costFn = trainer.createCostFn(X,y)
  # Run Cost Function and retrieve gradient
  wsize = sum([w.size for w in nnet.weights])
  wvec = np.sin(3*np.arange(-1,1,2.0/wsize))
  cst,grd = costFn(wvec)
  # Run computeNumericalGradient with the network costs and weights
  cgrd = computeNumericalGradient(costFn,wvec)

  # for i in xrange(len(cgrd)):
  #   print "%10.6f %10.6f" % (cgrd[i], grd[i])
  for i in range(len(cgrd)):
    np.testing.assert_array_almost_equal(cgrd[i], grd[i], decimal = 3,
      err_msg = "The output grd at index %d should be \n %r \n != %r" % (i, cgrd[i], grd[i]))

def computeNumericalGradient(costFn, weights, e=0.000001):
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
"""
