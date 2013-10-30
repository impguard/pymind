""" Package of common error functions.

An error function is a static object that has two generalized class methods calc and grad. The error
function takes two arguments, namely the hypothesis and the expectation, and returns either the
error if calc is called, or the gradient of the error function if grad is called.

Any subclass of an error function can implement the methods _calc and _grad to perform a specific
error function on the passed hypothesis and expected values. Add the subclass to the errfn module
using the built in function add with an associated name. This function will be accessable anywhere
using the built in function get anywhere within pymind.

The error function always transforms any input into a numpy matrix and performs element-wise
operations, so the dimensions of the hypothesis and the expectation should match.
"""

import numpy as np
from util import assertType

fn_list = dict()
def get(name):
  assertType("errfn.get", "name", name, str)
  assert name in fn_list, "(errfn) %s cannot be found." % name
  return fn_list[name]

def add(name, fn):
  assertType("errfn.add", "name", name, str)
  fn_list[name] = fn

def contains(name):
  assertType("errfn.contains", "name", name, str)
  return name in fn_list

def getFnNames():
  return fn_list.keys()

class _errfn(object):
  """ Abstract factory base class for any generalized error function."""
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
  """ The squared error function.

  This error function returns the squared error: (1/2)(h - y)^2
  The gradient is also easily calculated as: (h - y)
  """
  @classmethod
  def _calc(cls, h, y):
    return 0.5 * np.power(h-y, 2)

  @classmethod
  def _grad(cls, h, y):
    return h-y

class logitError(_errfn):
  """ The logit error function.

  This error function returns the logit error: -ylog(h) - (1-y)log(1-h)
  """
  @classmethod
  def _calc(cls, h, y):
    return -np.multiply(y, np.log(h)) - np.multiply(1-y, np.log(1-h))

  @classmethod
  def _grad(cls, h, y):
    return np.divide(h-y, np.multiply(h, 1-h))

add("squaredError", squaredError)
add("logitError", logitError)
