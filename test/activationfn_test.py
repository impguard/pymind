import numpy as np
from pymind.activationfn import sigmoid

# Setup variables
v1 = 3
v2 = np.array([[1, 2, 3]]).T
v3 = np.matrix([1, 2, 3]).T

def testSigmoid():
  # Test that sigmoid calculation works for numbers
  result1 = sigmoid.calc(v1)
  assert type(result1) is np.matrix, "sigmoid.calc(3) should return a np.matrix: %s" % type(result1)
  np.testing.assert_approx_equal(result1.item(0), 0.952574, significant = 5,
    err_msg = "sigmoid.calc(3) should approximately return a value of 0.952574: %f" % result1.item(0))

  # Test that sigmoid calculation works for arrays and matrices and outputs the same thing
  result1 = sigmoid.calc(v2)
  result2 = sigmoid.calc(v3)
  np.testing.assert_array_equal(result1, result2,
    err_msg = "sigmoid.calc should be returning the same result for two identical vectors of different types")

def testSigmoidGrad():
  # Test that sigmoid gradient works for an input of 0
  result1 = sigmoid.grad(0)
  np.testing.assert_approx_equal(result1.item(0), 0.25, significant = 2,
    err_msg = "sigmoid.grad(0) should approximately return a value of 0.25: %f" % result1.item(0))

def testSigmoidAccuracy():
  # Testing sigmoid accuracy to 10 sig. digits
  v1 = np.matrix([0.567, 0.1451]).T
  o1 = np.matrix([0.638070653034863, 0.536211489194633]).T
  np.testing.assert_array_almost_equal(sigmoid.calc(v1), o1, decimal = 10,
    err_msg = "The output sigmoid.calc(v1) should be almost equal to %r: %r" % (o1, v1))

  # Testing sigmoid gradient accuracy to 10 sig. digits
  v2 = np.matrix([5, 2.7213]).T
  o2 = np.matrix([0.00664805667079003, 0.0579177681881207]).T
  np.testing.assert_array_almost_equal(sigmoid.grad(v2), o2, decimal = 10,
    err_msg = "The output sigmoid.grad(v1) should be almost equal to %r: %r" % (o2, v2))
