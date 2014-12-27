import numpy as np
from pymind.datautil import *
from os import popen

ds_orig = {"y": np.matrix([[1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
	[0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1]]),
	"X": np.matrix([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
	[1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1],
	[0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
	[0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0]])}

params = {
	"input_units": 5,
	"output_units": 3,
	"hidden_units": [4, 4, 3],
	"activationfn": [identity, sigmoid, sigmoid, sigmoid, sigmoid],
	"bias": False
}
nnet00 = NeuralNetwork(params)

__check_shape = lambda s,t : (s.shape == t) if type(t) is tuple else (s.shape == t.shape)
__check_diff = lambda s,t : (s - t).sum() == 0

def __check_nnet(nnet1, nnet2, msg):
	assert len(nnet1.weights) == len(nnet2.weights), msg + " (mismatched number of layers)"
	for i in xrange(len(nnet1.weights)):
		w1, w2 = nnet1.weights[i], nnet2.weights[i]
		assert __check_shape(w1, w2), msg + " (shapes of neural network weights are mismatched)"
		assert __check_diff(w1, w2), msg + " (values of neural network weights are mismatched)"

def __check_dsets(ds1,ds2,msg_X="Input failure",msg_y="Output failure"):
	assert (not (ds1["X"] != ds2["X"]).sum()),msg_X
	assert (not (ds1["y"] != ds2["y"]).sum()),msg_y

def test_builder():
	dsb = DatasetBuilder(4,2)
	dsb.add([1,1,0,0],[1,0])
	dsb.add([1,0,0,0],[0,0])
	dsb.add([1,0,0,1],[0,1])
	dsb.add([1,0,1,1],[0,1])
	dsb.add([1,1,1,1],[1,1])
	dsb.add([1,0,0,0],[0,0])
	dsb.add([0,0,0,0],[0,1])
	dsb.add([0,0,0,1],[1,1])
	dsb.add([0,1,0,1],[0,1])
	dsb.add([0,1,1,1],[1,1])
	dsb.add([0,1,1,0],[0,0])
	dsb.add([1,1,1,0],[0,1])
	ds = dsb.build()
	__check_dsets(ds_orig,ds,"Failure loading input data from .mat file at ds-4-2_00",
		"Failure loading output data from .mat file at ds-4-2_00")

def test_load_data_mat():
	ds_loaded = load_data("test/datasets/ds-4-2_00.mat")
	__check_dsets(ds_loaded,ds_orig,"Failure loading input data from .mat file at ds-4-2_00",
		"Failure loading output data from .mat file at ds-4-2_00")

def test_load_data_json():
	ds_loaded = load_data("test/datasets/ds-4-2_00",format="json")
	__check_dsets(ds_loaded,ds_orig,"Failure loading input data from .json file at ds-4-2_00",
		"Failure saving loading data from .json file at ds-4-2_00")

def test_save_data_json():
	save_data("test/datasets/ds-4-2_00",ds_orig,format="json")
	ds_loaded = load_data("test/datasets/ds-4-2_00.json")
	__check_dsets(ds_loaded,ds_orig,"Failure saving input data to .json file at ds-4-2_00",
		"Failure saving output data to .json file at ds-4-2_00")

def test_save_data_mat():
	save_data("test/datasets/ds-4-2_01",ds_orig,format="mat")
	ds_loaded = load_data("test/datasets/ds-4-2_01",format="mat")
	__check_dsets(ds_loaded,ds_orig,"Failure saving input data to .mat file at ds-4-2_01",
		"Failure saving output data to .mat file at ds-4-2_01")
	popen("rm -f test/datasets/ds-4-2_01.mat")

def test_load_nnet_json():
	nnet_loaded1 = load_neural_net("test/datasets/nnet-54433_00.json")
	nnet_loaded2 = load_neural_net("test/datasets/nnet-54433_00", "json")
	__check_nnet(nnet_loaded1, nnet_loaded2, "Failure loading neural net")

def test_save_nnet_json():
	save_neural_net("test/datasets/nnet-save-test.json", nnet00)
	nnet_loaded = load_neural_net("test/datasets/nnet-save-test", "json")
	__check_nnet(nnet_loaded, nnet00, "Failure saving neural net")
	popen("rm -f test/datasets/nnet-save-test.json")

def test_split_dataset1():
	X,y = ds_orig["X"],ds_orig["y"]
	s = split_data(X,y,3)
	assert len(s)==3, "Data should split into 3 groups, not %i" % len(s)
	assert __check_shape(s[0]["X"],(4,4)) and __check_shape(s[1]["X"],(4,4)) and\
		__check_shape(s[2]["X"],(4,4)), "Data should split evenly into ((4,4),(4,4),(4,4)), not\
		(%i,%i,%i)" % (s[0]["X"].shape,s[1]["X"].shape,s[2]["X"].shape)

def test_split_dataset2():
	X,y = ds_orig["X"],ds_orig["y"]
	s = split_data(X,y,[1,2])
	assert len(s)==2, "Data should split into 2 groups, not %i" % len(s)
	assert __check_shape(s[0]["X"],(4,4)) and __check_shape(s[1]["X"],(4,8)),\
		"Data should split evenly into ((4,4),(4,8)), not (%i,%i)" %\
		(s[0]["X"].shape,s[1]["X"].shape)
