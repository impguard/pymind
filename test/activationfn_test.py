import numpy as np
import pymind.activationfn as af
from pymind.activationfn import sigmoid, identity

def testGetPlugin1():
  """ Test that activationfn can properly get functions. """
  assert af.get("sigmoid") is sigmoid, "Sigmoid should be automatically added to activationfn."
  assert af.get("identity") is identity, "Identity should be automatically added to activationfn."
  assert af.contains("sigmoid"), "activationfn.contains('sigmoid') should return True."
  assert af.contains("identity"), "activationfn.contains('identity') should return True."

def testAddPlugin1():
  """ Test that activationfn can properly add functions. """
  af.add("random", 1)
  af.add("another", 2)

  assert af.get("random") is 1, "A function called random should be stored in activationfn."
  assert af.get("another") is 2, "A function called another should be stored in activationfn."
  assert af.contains("random"), "activationfn.contains('random') should return True."
  assert af.contains("another"), "activationfn.contains('another') should return True."

def testSigmoidCalc1():
  """ Test that sigmoid.calc works to 10 sig. digits. """
  num = 3
  result = sigmoid.calc(num)
  assert type(result) is np.matrix, "sigmoid.calc(3) should return a np.matrix"
  np.testing.assert_approx_equal(result.item(0), 0.952574126822433, significant = 10,
    err_msg = "sigmoid.calc(3) should approximately return a value of 0.952574126822433")

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
