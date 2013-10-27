""" Package of common metric functions, as well as combiner functions.

An metric function is a static object that takes a neural network and extracts
information (i.e. metrics) from the neural network using the function 'extract',
returning the information as any datatype.

A combiner function is a static object that takes a running result of calling
metric functions and the result of the latest call to a metric function, and
combines them, returning the resulting object.
"""

import numpy as np
from util import assertType

metric_list = dict()
def get_metric(name):
  assertType("metricfn.get_metric", "name", name, str)
  assert name in metric_list, "(metricfn) %s cannot be found." % name
  return metric_list[name]

def add_metric(name, fn):
  assertType("metricfn.add_metric", "name", name, str)
  metric_list[name] = fn

class _metricfn(object):
  @classmethod
  def extract(cls, nnet):
    return cls._extract(nnet)

  @classmethod
  def _extract(cls, nnet):
    raise Exception("_extract not implemented")

combiner_list = dict()
def get_combiner(name):
  assertType("metricfn.get_combiner", "name", name, str)
  assert name in metric_list, "(metricfn) %s cannot be found." % name
  return combiner_list[name]

def add_combiner(name, fn):
  assertType("metricfn.add_combiner", "name", name, str)
  combiner_list[name] = fn

class _combinerfn(object):
  @classmethod
  def reduce(cls, total, res):
    return cls._reduce

  @classmethod
  def _reduce(cls, total, res):
    raise Exception("_reduce not implemented")

class ListCombiner(_combinerfn):
  @classmethod
  def _reduce(cls, total, res):
    if total is None:
      return [res,]
    else:
      # list.append would mutate total. Is this what we want? 
      return total + [res,]

add_combiner("list_combiner",ListCombiner)