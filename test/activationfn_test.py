import numpy as np
from pymind.activationfn import sigmoid

def testSigmoidCalc1():
  """ Test that sigmoid.calc works to 10 sig. digits. """
  num = 3
  result = sigmoid.calc(num)
  assert type(result) is np.matrix, "sigmoid.calc(3) should return a np.matrix"
  np.testing.assert_approx_equal(result.item(0), 0.952574126822433, significant = 10,
    err_msg = "sigmoid.calc(3) should approximately return a value of 0.0.952574126822433")

def testSigmoidCalc2():
  """ Testing sigmoid.calc to 10 sig. digits. """
  v = np.matrix([0.567, 0.1451])
  o = np.matrix([0.638070653034863, 0.536211489194633])
  np.testing.assert_array_almost_equal(sigmoid.calc(v), o, decimal = 10,
    err_msg = "The output sigmoid.grad(%r) should be almost equal to %r" % (v, o))

def testSigmoidArguments1():
  """ Test that sigmoid.calc works for arrays and matrices and outputs the same thing. """
  v1 = np.matrix(np.arange(3))
  v2 = np.arange(3)
  result1 = sigmoid.calc(v1)
  result2 = sigmoid.calc(v2)
  np.testing.assert_array_equal(result1, result2,
    err_msg = "sigmoid.calc should be returning the same \
    result for two identical vectors of different types")

def testSigmoidGrad1():
  """ Test that sigmoid.grad works for an input of 0. """
  result1 = sigmoid.grad(0)
  np.testing.assert_approx_equal(result1.item(0), 0.25, significant = 10,
    err_msg = "sigmoid.grad(0) should approximately return a value of 0.25")

def testSigmoidGrad2():
  """ Testing sigmoid.grad to 10 sig. digits. """
  v = np.matrix([5, 2.7213])
  o = np.matrix([0.00664805667079003, 0.0579177681881207])
  np.testing.assert_array_almost_equal(sigmoid.grad(v), o, decimal = 10,
    err_msg = "The output sigmoid.grad(%r) should be almost equal to %r" % (v, o))
