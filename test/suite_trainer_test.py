import pymind as pm
import pymind.activationfn as af
import pymind.errfn as ef
import numpy as np
from pymind.components.nnbuilder import Builder, DEFAULT
from pymind.training import *
from pymind.metricfn import *

X = np.matrix([
	[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
	[1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1],
	[0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
	[0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0]])

y = np.matrix([
	[1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
	[0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1]])

def trivial_metric(nnet, res):
	return 1

def accumulate_combinator(total, res):
	if total is None:
		return res
	else:
		return total + res

def error_metric(nnet, res):
	return res.fun if res.status==0 else None

def testListCombiner1():
	b = Builder()
	b.set(bias=[True, False, True, False])
	b.set(X=X)
	b.set(y=y)
	b.set(layer_units=[[4,4,2],[4,4,2],[4,6,2],[4,6,2]])
	suites = b.build()
	res = train_suites(suites,trivial_metric)
	assert res == [1,1,1,1], "List combiner should return [1,1,1,1], not " + str(res)

def testAccumulationCombiner2():
	b = Builder()
	b.set(bias=[True, False, True, False])
	b.set(X=X)
	b.set(y=y)
	b.set(layer_units=[[4,4,2],[4,4,2],[4,6,2],[4,6,2]])
	suites = b.build()
	res = train_suites(suites,trivial_metric,combiner=accumulate_combinator)
	assert res == 4, "Accumulation combinator should return 4, not " + str(res)

def testErrorMetric1():
	b = Builder()
	b.set(bias=[True, False, True, False])
	b.set(X=X)
	b.set(y=y)
	b.set(layer_units=[[4,4,2],[4,4,2],[4,6,2],[4,6,2]])
	suites = b.build()
	res = np.array(train_suites(suites,error_metric))
	assert (res < 0.3).sum()==4, "Training error too high: " + str(res)

def testErrorMetric2():
	b = Builder()
	b.set(bias=True)
	b.set(X=X)
	b.set(y=y)
	b.set(layer_units=[[4,1,2],[4,3,2],[4,5,2],[4,7,2],[4,9,2],[4,11,2]])
	suites = b.build()
	res = np.array(train_suites(suites,error_metric))
	assert (res < 0.3).sum()==6, "Training error too high: " + str(res)
