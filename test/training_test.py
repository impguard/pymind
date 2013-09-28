import numpy as np
import pymind
from pymind import NeuralNetwork
from pymind.activationfn import identity, sigmoid
from pymind.errfn import logitError, squaredError
from pymind.util import generate_matrix, unroll_matrices, reshape_vector, create_minimizer

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
  for i in xrange(numGrad.size):
    perturb[i] = e
    pos = fn(x + perturb)
    neg = fn(x - perturb)
    numGrad[i] = (pos - neg) / (2. * e)
    perturb[i] = 0

  return numGrad

def testCostAccuracy1():
  """ Tests accuracy of cost calculated by a costfn created by create_costfn to 10 sig. digits.

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

  # Create cost function parameters
  X = generate_matrix(10, 3)
  y = generate_matrix(6, 3)
  learn_rate = 0
  errfn = logitError

  # Create cost function
  costfn = pymind.create_costfn(nnet, X, y, learn_rate, errfn)

  # Generate input weights
  weights = [generate_matrix(weight.shape[0], weight.shape[1]) for weight in nnet.weights]

  # Run cost function
  cost, grad = costfn(weights)
  actual = 5.40419468413414 # Calculated with Octave

  np.testing.assert_almost_equal(cost, actual, decimal=10,
    err_msg = "Cost function created by create_costfn does not calculate cost accurately enough.")

def testCostAccuracy2():
  """ Tests accuracy of cost calculated by a costfn created by create_costfn to 10 sig. digits.

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

  # Create cost function parameters
  X = generate_matrix(15, 10)
  y = generate_matrix(5, 10)
  learn_rate = 1
  errfn = logitError

  # Create cost function (don't compute gradient)
  costfn = pymind.create_costfn(nnet, X, y, learn_rate, errfn, False)

  # Generate input weights
  weights = [generate_matrix(weight.shape[0], weight.shape[1]) for weight in nnet.weights]

  # Run cost function
  cost = costfn(weights)
  actual = 9.00036890107429 # Calculated with Octave

  np.testing.assert_almost_equal(cost, actual, decimal=10,
    err_msg = "Cost function created by create_costfn does not calculate cost accurately enough.")

def testGradientAccuracy1():
  """ Tests accuracy of gradient calculated by a costfn created by create_costfn to 10 sig. digits.

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

  # Create cost function parameters
  X = generate_matrix(10, 3)
  y = generate_matrix(6, 3)
  learn_rate = 0
  errfn = logitError

  # Create cost function
  costfn = pymind.create_costfn(nnet, X, y, learn_rate, errfn)

  # Generate input weights
  weights = [generate_matrix(weight.shape[0], weight.shape[1]) for weight in nnet.weights]

  # Run cost function
  cost, grad = costfn(weights)

  # Simple unrolled cost function to use in computeNumericalGradient
  costfn = pymind.create_costfn(nnet, X, y, learn_rate, errfn, False)
  dimensions = [weight.shape for weight in nnet.weights]
  unrolledCostfn = lambda w: costfn(reshape_vector(w, dimensions))

  testGrad = computeNumericalGradient(unrolledCostfn, unroll_matrices(weights))

  np.testing.assert_array_almost_equal(unroll_matrices(grad), testGrad, decimal = 9,
      err_msg = "Cost function created by create_costfn does not calculate gradient accurately enough.")

def testGradientAccuracy2():
  """ Tests accuracy of gradient calculated by a costfn created by create_costfn to 10 sig. digits.

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

  # Create cost function parameters
  X = generate_matrix(15, 100)
  y = generate_matrix(10, 100)
  learn_rate = 1.5
  errfn = logitError

  # Create cost function
  costfn = pymind.create_costfn(nnet, X, y, learn_rate, errfn)

  # Generate input weights
  weights = [generate_matrix(weight.shape[0], weight.shape[1]) for weight in nnet.weights]

  # Run cost function
  cost, grad = costfn(weights)

  # Simple unrolled cost function to use in computeNumericalGradient
  costfn = pymind.create_costfn(nnet, X, y, learn_rate, errfn, False)
  unrolledCostfn = lambda w: costfn(reshape_vector(w, nnet.dimensions))

  testGrad = computeNumericalGradient(unrolledCostfn, unroll_matrices(weights))

  np.testing.assert_array_almost_equal(unroll_matrices(grad), testGrad, decimal = 9,
      err_msg = "Cost function created by create_costfn does not calculate gradient accurately enough.")

def testOR():
  """ Tests if pymind can train OR function. """
  params = {
    "input_units": 2,
    "output_units": 1,
    "activationfn": [identity, sigmoid],
    "bias": True
  }
  nnet = NeuralNetwork(params)

  X = np.matrix([[0,0,1,1],
                 [0,1,0,1]])
  y = np.matrix([[0,1,1,1]])

  minimizer = create_minimizer(iterations = 25, display = False)

  # Run 10 times and pick result with lowest error
  result = pymind.train(nnet, X, y, 0.0001, logitError, minimizer)

  h = nnet.activate(X)
  h = np.where(h > 0.99, np.ones(h.shape), h)
  h = np.where(h < 0.01, np.zeros(h.shape), h)

  np.testing.assert_array_equal(h, y, err_msg = "Learning OR failed")

def testAND():
  """ Tests if pymind can train AND function. """
  params = {
    "input_units": 2,
    "output_units": 1,
    "activationfn": [identity, sigmoid],
    "bias": True
  }
  nnet = NeuralNetwork(params)

  X = np.matrix([[0,0,1,1],
                 [0,1,0,1]])
  y = np.matrix([[0,0,0,1]])

  minimizer = create_minimizer(iterations = 25, display = False)

  # Run 10 times and pick result with lowest error
  result = pymind.train(nnet, X, y, 0.0001, logitError, minimizer)

  h = nnet.activate(X)
  h = np.where(h > 0.99, np.ones(h.shape), h)
  h = np.where(h < 0.01, np.zeros(h.shape), h)

  np.testing.assert_array_equal(h, y, err_msg = "Learning AND failed")

def testNAND():
  """ Tests if pymind can train NAND function. """
  params = {
    "input_units": 2,
    "output_units": 1,
    "activationfn": [identity, sigmoid],
    "bias": True
  }
  nnet = NeuralNetwork(params)

  X = np.matrix([[0,0,1,1],
                 [0,1,0,1]])
  y = np.matrix([[1,0,0,0]])

  minimizer = create_minimizer(iterations = 25, display = False)

  # Run 10 times and pick result with lowest error
  result = pymind.train(nnet, X, y, 0.0001, logitError, minimizer)

  h = nnet.activate(X)
  h = np.where(h > 0.99, np.ones(h.shape), h)
  h = np.where(h < 0.01, np.zeros(h.shape), h)

  np.testing.assert_array_equal(h, y, err_msg = "Learning NAND failed")

def testXOR():
  """ Tests if pymind can train XOR function. """
  params = {
    "input_units": 2,
    "hidden_units": 2,
    "output_units": 1,
    "activationfn": [identity, sigmoid, sigmoid],
    "bias": True
  }
  nnet = NeuralNetwork(params)

  X = np.matrix([[0,0,1,1],
                 [0,1,0,1]])
  y = np.matrix([[1,0,0,1]])

  minimizer = create_minimizer(iterations = 100, display = False)

  # Run 10 times and pick result with lowest error
  result = pymind.train(nnet, X, y, 0.001, logitError, minimizer)

  h = nnet.activate(X)
  h = np.where(h > 0.9, np.ones(h.shape), h)
  h = np.where(h < 0.1, np.zeros(h.shape), h)

  np.testing.assert_array_equal(h, y, err_msg = "Learning XOR failed")

def testPolynomial1():
  """ Tests if pymind can train degree 3 polynomial. """
  params = {
    "input_units": 3,
    "output_units": 1,
    "activationfn": [identity, identity],
    "bias": True
  }
  nnet = NeuralNetwork(params)

  polynomial = lambda x: 1 + 3 * x - np.power(x, 2) + np.power(x, 3)

  row_X = np.matrix(np.arange(50))
  X = np.vstack((row_X, np.power(row_X, 2), np.power(row_X, 3)))
  y = polynomial(row_X)

  minimizer = create_minimizer(iterations = 50, display = True)

  # Run 10 times and pick result with lowest error
  result = pymind.train(nnet, X, y, 0, squaredError, minimizer, iterations = 5)

  h = nnet.activate(X)

  check = np.where(np.abs(h - y) > 1, np.ones(h.shape), np.zeros(h.shape))
  np.testing.assert_array_equal(check, np.zeros(h.shape), err_msg = "Learning polynomial2 failed")

def testPolynomial2():
  """ Tests if pymind can train degree 3 polynomial using regularization. """
  params = {
    "input_units": 4,
    "output_units": 1,
    "activationfn": [identity, identity],
    "bias": True
  }
  nnet = NeuralNetwork(params)

  polynomial = lambda x: 1 + 3 * x - np.power(x, 2) + np.power(x, 3)

  row_X = np.matrix(np.arange(50))
  X = np.vstack((row_X, np.power(row_X, 2), np.power(row_X, 3), np.power(row_X, 4)))
  y = polynomial(row_X)

  minimizer = create_minimizer(iterations = 50, display = True)

  # Run 10 times and pick result with lowest error
  result = pymind.train(nnet, X, y, 0.001, squaredError, minimizer, iterations = 1)

  h = nnet.activate(X)

  check = np.where(np.abs(h - y) > 1, np.ones(h.shape), np.zeros(h.shape))
  np.testing.assert_array_equal(check, np.zeros(h.shape), err_msg = "Learning polynomial2 failed")
