""" Package of common activation functions.

An activation function is a static object that has two generalized class methods calc and grad. The
activation function takes one argument and returns either the activation if calc is called or the
gradient of the activation function if grad is called.

Any sublcass of an activation function can implement the methods _calc and _grad to perform a
specific activation function on the passed argument.

The activation function always transforms any input into a numpy matrix and performs element-wise
operations.
"""

import numpy as np

class _activationfn(object):
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

class sigmoid(_activationfn):
  """ Performs the sigmoid activation function. """
  @classmethod
  def _calc(cls, v):
    e = np.matrix(np.ones(v.shape) * np.e)
    return 1.0 / (1.0 + np.power(e, -v))

  @classmethod
  def _grad(cls, v):
    val = cls.calc(v)
    return np.multiply(val, 1-val)

class identity(_activationfn):
  """ Performs the simple identity activation function. """
  @classmethod
  def _calc(cls, v):
    return v

  @classmethod
  def _grad(cls, v):
    return 1
