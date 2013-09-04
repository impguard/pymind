import numpy as np

class _errfn(object):
  @classmethod
  def calc(cls, h, y):
    # Always cast input to a matrix
    h = np.matrix(h)
    y = np.matrix(y)
    return cls._calc(h, y)

  @classmethod
  def grad(cls, h, y):
    h = np.matrix(h)
    y = np.matrix(y)
    return cls._grad(h, y)

  @classmethod
  def _calc(cls, h, y):
    raise Exception("_calc not implemented")

  @classmethod
  def _grad(cls, h, y):
    raise Exception("_grad not implemented")

class squaredError(_errfn):
  @classmethod
  def _calc(cls, h, y):
    return 0.5 * np.power(h-y, 2)

  @classmethod
  def _grad(cls, h, y):
    return h-y

class logitError(_errfn):
  @classmethod
  def _calc(cls, h, y):
    return -np.multiply(y, np.log(h)) - np.multiply(1-y, np.log(1-h))

  @classmethod
  def _grad(cls, h, y):
    return np.divide(h-y, np.multiply(h, 1-h))
