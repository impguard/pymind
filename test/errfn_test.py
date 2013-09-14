import numpy as np
from pymind.errfn import squaredError, logitError

def testSquaredErrorAccuracy():
  # Testing squared error function accuracy to 10 sig. digits
  h1 = np.matrix([1.23, 4.56, 7.89]).T
  y1 = np.matrix([3.45, 10.56, 1.14]).T
  err = squaredError.calc(h1, y1)
  o1 = np.matrix([2.46420000000000, 18.00000000000001, 22.78125000000000]).T

  np.testing.assert_array_almost_equal(err, o1, decimal = 10,
    err_msg = "Squared error function is not accurate enough for (h1, y1).")

  # Testing squared error function accuracy to 10 sig. digits
  h2 = np.matrix([1.345566661, 11235.12315123, 8.12312152]).T
  y2 = np.matrix([312.456123, 12151.12345123, 1.2141255]).T
  err = squaredError.calc(h2, y2)
  o2 = np.matrix([4.83948891327810e4, 4.19528274800045e5, 2.38671130021879e1]).T

  np.testing.assert_array_almost_equal(err, o2, decimal = 9,
    err_msg = "Squared error function is not accurate enough for (h2, y2).")

  # Use previous values of h1, y1, h2, y2 to test squared error
  grad1 = squaredError.grad(h1, y1)
  grad2 = squaredError.grad(h2, y2)
  o3 = np.matrix([-2.22000000000000, -6.00000000000000, 6.75000000000000]).T
  o4 = np.matrix([-311.11055633899997, -916.00029999999970, 6.90899602000000]).T

  np.testing.assert_array_almost_equal(np.vstack((grad1, grad2)), np.vstack((o3, o4)), decimal = 10,
    err_msg = "Squared error gradient is not accurate enough for (h1, y1) and (h2, y2).")


def testLogitErrorAccuracy():
  # Testing logit error function to 10 sig. digits
  h1 = np.matrix([0.542, 0.341, 0.532]).T
  y1 = np.matrix([1, 0.921, 0.0123]).T
  err = logitError.calc(h1, y1)
  grad = logitError.grad(h1, y1)
  o1 = np.matrix([0.612489277542491, 1.023824358178320, 0.757710428185375]).T
  o2 = np.matrix([-1.84501845018450, -2.58100116145052, 2.08734978471821]).T

  np.testing.assert_array_almost_equal(err, o1, decimal = 10,
    err_msg = "Logit error function is not accurate enough for (h1, y1).")
  np.testing.assert_array_almost_equal(grad, o2, decimal = 10,
    err_msg = "Logit error gradient is not accurate enough for (h1, y1).")
