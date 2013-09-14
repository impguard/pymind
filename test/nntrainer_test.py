import numpy as np
from numpy.random import random as nprandom
from pymind.components import NeuralNetwork, NNTrainer
from pymind.activationfn import *
from pymind.errfn import *

def generateMatrix(rows, columns, fn = np.sin):
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
  return fn(np.matrix(matrix))

def computeNumericalGradient(fn, x, e = 1e-4):
  """ Computes the gradient of a function using "finite differences".

  Arguments:
  fn -- the function to compute the numerical gradient of
  x -- the value to to evaluate the numerical gradient at
  e -- the perturbation being used to focus the calculation (default: 1e-4)
  Returns:
  The gradient of fn at x

  Note: x should be a column vector representing the input to the fn.
  """
  numGrad = np.matrix(np.zeros(x.shape))
  perturb = np.matrix(np.zeros(x.shape))
  for i in range(numGrad.size):
    perturb[i] = e
    pos = fn(x + e)
    neg = fn(x - e)
    numGrad[i] = (pos - neg) / (2. * e)
    perturb[i] = 0

  return numGrad

def testCostAccuracy1():
  """ Tests accuracy of cost function in the nntrainer class to 10 sig. digits.

  Turns off regularization by setting the learning rate to 0.
  """
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
  weights = [generateMatrix(weight.shape[0], weight.shape[1]) for weight in nnet.weights]

  # Run cost function
  cost, grad = costfn(weights)
  actual = 5.40419468413414 # Calculated with Octave

  np.testing.assert_almost_equal(cost, actual, decimal=10,
    err_msg = "Trainer cost function not accurate enough.")

def testCostAccuracy2():
  """ Tests accuracy of cost function in the nntrainer class to 10 sig. digits.

  Turns on regularization by setting the learning rate to 1.
  """
  params = {
    "input_units": 15,
    "output_units": 5,
    "hidden_units": 10,
    "activationfn": [identity, sigmoid, sigmoid],
    "bias": True
  }
  nnet = NeuralNetwork(params)
  trainer = NNTrainer(nnet)

  # Create cost function parameters
  X = generateMatrix(15, 10)
  y = generateMatrix(5, 10)
  learn_rate = 1
  errfn = logitError

  # Create cost function (don't compute gradient)
  costfn = trainer.createCostfn(X, y, learn_rate, errfn, False)

  # Generate input weights
  weights = [generateMatrix(weight.shape[0], weight.shape[1]) for weight in nnet.weights]

  # Run cost function
  cost = costfn(weights)
  actual = 9.00036890107429 # Calculated with Octave

  np.testing.assert_almost_equal(cost, actual, decimal=10,
    err_msg = "Trainer cost function not accurate enough.")

def testGradientAccuracy1():
  """ Tests accuracy of the trainer's gradient calculation to 10 sig. digits.

  Turns off regularization by setting learn_rate to 0.
  """
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
  weights = [generateMatrix(weight.shape[0], weight.shape[1]) for weight in nnet.weights]

  # Run cost function
  cost, grad = costfn(weights)

  # Simple unrolled cost function to use in computeNumericalGradient
  costfn = trainer.createCostfn(X, y, learn_rate, errfn, False)
  unrolledCostfn = lambda w: costfn(trainer.reshapeWeights(w))

  testGrad = computeNumericalGradient(unrolledCostfn, trainer.unrollWeights(weights))
  # print trainer.unrollWeights(grad)

  np.testing.assert_array_almost_equal(trainer.unrollWeights(grad), testGrad, decimal = 10,
      err_msg = "Trainer gradient calculation is not accurate enough.")

def testUnrollReshapeWeights():
  """ Tests if trainer weight unrolling and reshaping works properly. """
  params = {
    "input_units": 10,
    "output_units": 6,
    "hidden_units": 8,
    "activationfn": [identity, sigmoid, sigmoid],
    "bias": True
  }
  nnet = NeuralNetwork(params)
  trainer = NNTrainer(nnet)

  unrolledWeights = trainer.unrollWeights(nnet.weights)
  reshapedWeights = trainer.reshapeWeights(unrolledWeights)

  for i in range(len(reshapedWeights)):
    np.testing.assert_array_equal(reshapedWeights[i], nnet.weights[i],
      err_msg = "The reshapedWeight at index %d is not the same as the \
      same weight in the neural network." % i)
