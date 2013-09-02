import numpy as np

class _activationFn(object):
  @classmethod
  def calc(cls, v):
    # Always cast input to a matrix
    v = np.matrix(v)
    return cls._calc(v)

  @classmethod
  def grad(cls, v):
    v = np.matrix(v)
    return cls._grad(v)

  @classmethod
  def _calc(cls, v):
    raise Exception("_calc not implemented")

  @classmethod
  def _grad(cls, v):
    raise Exception("_grad not implemented")

class sigmoid(_activationFn):
  @classmethod
  def _calc(cls, v):
    e = np.matrix(np.ones(v.shape) * np.e)
    return 1.0 / (1.0 + np.power(e, -v))

  @classmethod
  def _grad(cls, v):
    val = cls.calc(v)
    return np.multiply(val, 1-val)

class identity(_activationFn):
  @classmethod
  def _calc(cls, v):
    return v

  @classmethod
  def _grad(cls, v):
    return v
