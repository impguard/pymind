""" Package of common metric functions, as well as combiner functions.

An metric is a function that takes a neural network and extracts information (i.e. metrics) from the
neural network, returning the information as any datatype.

A combiner function is a function that takes a running result of calling metric functions and the
result of the latest call to a metric function, and combines them, returning the resulting object.
"""

import numpy as np
from util import assertType

_metrics = dict()
def get_metric(name):
  """ Gets the metric function corresponding to this name. If the name corresponds to no function,
  raises an exception.

  Arguments:
  name -- a string representing the name of this metric
  Returns:
  a metric mapped from the given name
  """
  assertType("metricfn.get_metric", "name", name, str)
  assert name in _metrics, "(metricfn) %s cannot be found." % name
  return _metrics[name]

def set_metric(name, fn):
  """ Sets the metric function corresponding using this name. Overwrites the function if the name
  already maps to a function.

  Arguments:
  name -- a string representing the name of this metric
  fn -- a function that takes a NeuralNetwork and returns some value derived from the NeuralNetwork
  """
  assertType("metricfn.set_metric", "name", name, str)
  _metrics[name] = fn

_combiners = dict()
def get_combiner(name):
  """ Gets the combiner function corresponding to this name. If the name corresponds to no function,
  raises an exception.

  Arguments:
  name -- a string representing the name of this combiner
  Returns:
  a combiner mapped from the given name
  """
  assertType("metricfn.get_combiner", "name", name, str)
  assert name in _combiners, "(metricfn) %s cannot be found." % name
  return _combiners[name]

def set_combiner(name, fn):
  """ Sets the combiner function corresponding using this name. If the name already maps to a
  function, overwrites the function.

  Arguments:
  name -- a string representing the name of this combiner
  fn -- a function that takes a total and a result and returns the combination of the two
  """
  assertType("metricfn.set_combiner", "name", name, str)
  _combiners[name] = fn

def __list_combiner(total, res):
  """ Returns total concatenated with res. If total is None, returns res as a single element list.
  This is the default combiner function.
  """
  if total is None:
    return [res,]
  else:
    # using list.append would mutate total. Is this what we want? 
    return total + [res,]

set_combiner("list_combiner",__list_combiner)
