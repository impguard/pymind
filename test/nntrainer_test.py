import numpy as np
from numpy.random import random as nprandom
from pymind.components import NeuralNetwork, NNTrainer
from pymind.activationfn import sigmoid, identity
from pymind.errfn import squaredError, logitError
from scipy.optimize import minimize

def createMinimizer(method = 'BFGS', tolerance = 1e-3, iterations = 50, display = False):
  """ Creates a minimizer using scipy.optimize optimization functions. """
  def minimizer(fn, initial_guess):
    return minimize(fn, initial_guess,
      method = method,
      jac = True,
      tol = tolerance,
      options = {
        'maxiter': iterations,
        'disp': display
      })
  return minimizer

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
    pos = fn(x + perturb)
    neg = fn(x - perturb)
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

  np.testing.assert_array_almost_equal(trainer.unrollWeights(grad), testGrad, decimal = 9,
      err_msg = "Trainer gradient calculation is not accurate enough.")

def testGradientAccuracy2():
  """ Tests accuracy of the trainer's gradient calculation to 10 sig. digits.

  Turns on regularization by setting learn_rate to 1.
  """
  params = {
    "input_units": 15,
    "output_units": 10,
    "hidden_units": [12, 12],
    "activationfn": [identity, sigmoid, sigmoid, sigmoid],
    "bias": True
  }
  nnet = NeuralNetwork(params)
  trainer = NNTrainer(nnet)

  # Create cost function parameters
  X = generateMatrix(15, 100)
  y = generateMatrix(10, 100)
  learn_rate = 1.5
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

  np.testing.assert_array_almost_equal(trainer.unrollWeights(grad), testGrad, decimal = 9,
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

def testOR():
  """ Tests if trainer can train OR function. """
  params = {
    "input_units": 2,
    "output_units": 1,
    "activationfn": [identity, sigmoid],
    "bias": True
  }
  nnet = NeuralNetwork(params)
  trainer = NNTrainer(nnet)

  X = np.matrix([[0,0,1,1],
                 [0,1,0,1]])
  y = np.matrix([[0,1,1,1]])

  minimizer = createMinimizer(iterations = 25, display = False)

  # Run 10 times and pick result with lowest error
  result = trainer.train(X, y, 0.0001, logitError, minimizer)

  z, a = nnet.feedForward(X)
  h = a[-1]
  h = np.where(h > 0.99, np.ones(h.shape), h)
  h = np.where(h < 0.01, np.zeros(h.shape), h)

  np.testing.assert_array_equal(h, y, err_msg = "Learning OR failed")

def testAND():
  """ Tests if trainer can train AND function. """
  params = {
    "input_units": 2,
    "output_units": 1,
    "activationfn": [identity, sigmoid],
    "bias": True
  }
  nnet = NeuralNetwork(params)
  trainer = NNTrainer(nnet)

  X = np.matrix([[0,0,1,1],
                 [0,1,0,1]])
  y = np.matrix([[0,0,0,1]])

  minimizer = createMinimizer(iterations = 25, display = False)

  # Run 10 times and pick result with lowest error
  result = trainer.train(X, y, 0.0001, logitError, minimizer)

  z, a = nnet.feedForward(X)
  h = a[-1]
  h = np.where(h > 0.99, np.ones(h.shape), h)
  h = np.where(h < 0.01, np.zeros(h.shape), h)

  np.testing.assert_array_equal(h, y, err_msg = "Learning AND failed")

def testNAND():
  """ Tests if trainer can train NAND function. """
  params = {
    "input_units": 2,
    "output_units": 1,
    "activationfn": [identity, sigmoid],
    "bias": True
  }
  nnet = NeuralNetwork(params)
  trainer = NNTrainer(nnet)

  X = np.matrix([[0,0,1,1],
                 [0,1,0,1]])
  y = np.matrix([[1,0,0,0]])

  minimizer = createMinimizer(iterations = 25, display = False)

  # Run 10 times and pick result with lowest error
  result = trainer.train(X, y, 0.0001, logitError, minimizer)

  z, a = nnet.feedForward(X)
  h = a[-1]
  h = np.where(h > 0.99, np.ones(h.shape), h)
  h = np.where(h < 0.01, np.zeros(h.shape), h)

  np.testing.assert_array_equal(h, y, err_msg = "Learning NAND failed")

def testXOR():
  """ Tests if trainer can train XOR function. """
  params = {
    "input_units": 2,
    "hidden_units": 2,
    "output_units": 1,
    "activationfn": [identity, sigmoid, sigmoid],
    "bias": True
  }
  nnet = NeuralNetwork(params)
  trainer = NNTrainer(nnet)

  X = np.matrix([[0,0,1,1],
                 [0,1,0,1]])
  y = np.matrix([[1,0,0,1]])

  minimizer = createMinimizer(iterations = 100, display = False)

  # Run 10 times and pick result with lowest error
  result = trainer.train(X, y, 0.001, logitError, minimizer)

  z, a = nnet.feedForward(X)
  h = a[-1]
  h = np.where(h > 0.9, np.ones(h.shape), h)
  h = np.where(h < 0.1, np.zeros(h.shape), h)

  np.testing.assert_array_equal(h, y, err_msg = "Learning XOR failed")

def testPolynomial1():
  """ Tests if trainer can train degree 3 polynomial. """
  params = {
    "input_units": 3,
    "output_units": 1,
    "activationfn": [identity, identity],
    "bias": True
  }
  nnet = NeuralNetwork(params)
  trainer = NNTrainer(nnet)

  polynomial = lambda x: 1 + 3 * x - np.power(x, 2) + np.power(x, 3)

  row_X = np.matrix(np.arange(50))
  X = np.vstack((row_X, np.power(row_X, 2), np.power(row_X, 3)))
  y = polynomial(row_X)

  minimizer = createMinimizer(iterations = 50, display = True)

  # Run 10 times and pick result with lowest error
  result = trainer.train(X, y, 0, squaredError, minimizer, iterations = 5)

  z, a = nnet.feedForward(X)
  h = a[-1]

  check = np.where(np.abs(h - y) > 1, np.ones(h.shape), np.zeros(h.shape))
  np.testing.assert_array_equal(check, np.zeros(h.shape), err_msg = "Learning polynomial2 failed")

def testPolynomial2():
  """ Tests if trainer can train degree 3 polynomial using regularization. """
  params = {
    "input_units": 4,
    "output_units": 1,
    "activationfn": [identity, identity],
    "bias": True
  }
  nnet = NeuralNetwork(params)
  trainer = NNTrainer(nnet)

  polynomial = lambda x: 1 + 3 * x - np.power(x, 2) + np.power(x, 3)

  row_X = np.matrix(np.arange(50))
  X = np.vstack((row_X, np.power(row_X, 2), np.power(row_X, 3), np.power(row_X, 4)))
  y = polynomial(row_X)

  minimizer = createMinimizer(iterations = 50, display = True)

  # Run 10 times and pick result with lowest error
  result = trainer.train(X, y, 0.001, squaredError, minimizer, iterations = 1)

  z, a = nnet.feedForward(X)
  h = a[-1]

  check = np.where(np.abs(h - y) > 1, np.ones(h.shape), np.zeros(h.shape))
  np.testing.assert_array_equal(check, np.zeros(h.shape), err_msg = "Learning polynomial2 failed")
