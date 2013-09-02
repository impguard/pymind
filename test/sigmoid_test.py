import numpy as np
from pymind import *

# Setup variables
v1 = 3
v2 = np.array([[1, 2, 3]]).T
v3 = np.matrix([1, 2, 3]).T

def testSigmoid():
  # Test that sigmoid calculation works for numbers
  result1 = sigmoid(v1)
  assert type(result1) is np.matrix, "sigmoid(3) should return a np.matrix: %s" % type(result1)
  np.testing.assert_approx_equal(result1.item(0), 0.952574, significant = 5,
    err_msg = "sigmoid(3) should approximately return a value of 0.952574: %f" % result1.item(0))

  # Test that sigmoid calculation works for arrays and matrices and outputs the same thing
  result1 = sigmoid(v2)
  result2 = sigmoid(v3)
  np.testing.assert_array_equal(result1, result2,
    err_msg = "sigmoid() should be returning the same result for two identical vectors of different types")

def testSigmoidGrad():
  # Test that sigmoid gradient works for an input of 0
  result1 = sigmoidGrad(0)
  np.testing.assert_approx_equal(result1.item(0), 0.25, significant = 2,
    err_msg = "sigmoid(0) should approximately return a value of 0.25: %f" % result1.item(0))
