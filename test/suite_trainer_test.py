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

trivial_metric = lambda nnet, res : 1


def testTrainerBasic():
	b = Builder()
	b.set(bias=[True, False, True, False])
	b.set(X=X)
	b.set(y=y)
	b.set(layer_units=[[4,4,2],[4,4,2],[4,6,2],[4,6,2]])
	suites = b.build()
	res = train_suites(suites,trivial_metric)
	assert res == [1,1,1,1], "List combiner should return [1,1,1,1], not " + str(res)
