""" A set of utility functions.

Note: These methods work with numpy matrices. Any mention of a matrix is a np.matrix, and a vector
  is a one dimensional matrix.
"""

from mathutil import *

def assertType(fn, name, var, t):
  if type(var) is not t:
    raise TypeError("(%s) Expected %s to be %s." % (fn, name, t))
