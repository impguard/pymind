import numpy as np
import pymind.errfn as ef
from pymind.errfn import squaredError, logitError

def testGetPlugin1():
  """ Test that activationfn can properly get functions. """
  assert ef.get("squaredError") is squaredError, \
    "squaredError should be automatically added to activationfn."
  assert ef.get("logitError") is logitError, \
    "logitError should be automatically added to activationfn."

def testAddPlugin1():
  """ Test that activationfn can properly add functions. """
  ef.add("random", 1)
  ef.add("another", 2)

  assert ef.get("random") is 1, "A function called random should be stored in activationfn."
  assert ef.get("another") is 2, "A function called another should be stored in activationfn."

def testSquaredErrorCalc1():
  """ Testing squaredError.calc to 10 sig. digits. """
  h = np.matrix([1.23, 4.56, 7.89])
  y = np.matrix([3.45, 10.56, 1.14])
  err = squaredError.calc(h, y)
  o = np.matrix([2.46420000000000, 18.00000000000001, 22.78125000000000])

  np.testing.assert_array_almost_equal(err, o, decimal = 10,
    err_msg = "Squared error function is not accurate enough for (h, y).")

def testSquaredErrorCalc2():
  """ Testing squaredError.calc to 10 sig. digits. """
  h = np.matrix([1.345566661, 11235.12315123, 8.12312152])
  y = np.matrix([312.456123, 12151.12345123, 1.2141255])
  err = squaredError.calc(h, y)
  o = np.matrix([4.83948891327810e4, 4.19528274800045e5, 2.38671130021879e1])

  np.testing.assert_array_almost_equal(err, o, decimal = 9,
    err_msg = "Squared error function is not accurate enough for (h, y).")

def testSquaredErrorGrad1():
  """ Testing squaredError.grad to 10 sig. digits. """
  h1 = np.matrix([1.23, 4.56, 7.89])
  y1 = np.matrix([3.45, 10.56, 1.14])
  h2 = np.matrix([1.345566661, 11235.12315123, 8.12312152])
  y2 = np.matrix([312.456123, 12151.12345123, 1.2141255])
  grad1 = squaredError.grad(h1, y1)
  grad2 = squaredError.grad(h2, y2)
  o1 = np.matrix([-2.22000000000000, -6.00000000000000, 6.75000000000000])
  o2 = np.matrix([-311.11055633899997, -916.00029999999970, 6.90899602000000])

  np.testing.assert_array_almost_equal(np.vstack((grad1, grad2)), np.vstack((o1, o2)), decimal = 10,
    err_msg = "Squared error gradient is not accurate enough for (h1, y1) and (h2, y2).")

def testLogitErrorCostGrad1():
  """ Testing logitError.calc and logitError.grad to 10 sig. digits. """
  h = np.matrix([0.542, 0.341, 0.532])
  y = np.matrix([1, 0.921, 0.0123])
  err = logitError.calc(h, y)
  grad = logitError.grad(h, y)
  err_o = np.matrix([0.612489277542491, 1.023824358178320, 0.757710428185375])
  grad_o = np.matrix([-1.84501845018450, -2.58100116145052, 2.08734978471821])

  np.testing.assert_array_almost_equal(err, err_o, decimal = 10,
    err_msg = "Logit error function is not accurate enough for (h, y).")
  np.testing.assert_array_almost_equal(grad, grad_o, decimal = 10,
    err_msg = "Logit error gradient is not accurate enough for (h, y).")
