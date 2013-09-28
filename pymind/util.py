import numpy as np

def assertType(fn, name, var, t):
  if type(var) is not t:
    raise TypeError("(%s) Expected %s to be %s." % (fn, name, t))

