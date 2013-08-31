import numpy as np

# Two functions to either calculate the sigmoid function or its gradient given an input. Both functions accept a
# numeric quantity or a np.ndarray, but will always return a np.matrix. (With the same dimensions. For a number, the
# return value will be a 1x1 matrix)

def calc(v):
  # Always cast input to a matrix
  v = np.matrix(v)
  e = np.matrix(np.ones(v.shape) * np.e)

  return 1.0 / (1.0 + np.power(e, -v))

def grad(v):
  val = calc(v)
  return np.multiply(val, 1-val)
