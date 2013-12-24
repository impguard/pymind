import numpy as np
from pymind.datautil import *
from os import popen

ds_orig = {"y": np.matrix([[1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
	[0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1]]),
	"X": np.matrix([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
	[1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1],
	[0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
	[0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0]])}

def __check_datasets(ds1,ds2,msg_X="Input failure",msg_y="Output failure"):
	assert (not (ds1["X"] != ds2["X"]).sum()),msg_X
	assert (not (ds1["y"] != ds2["y"]).sum()),msg_y

__check_shape = lambda s,t : s.shape == t

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
	__check_datasets(ds_orig,ds,"Failure loading input data from .mat file at ds-4-2_00",
		"Failure loading output data from .mat file at ds-4-2_00")

def test_load_mat():
	ds_loaded = load_data("test/datasets/ds-4-2_00.mat")
	__check_datasets(ds_loaded,ds_orig,"Failure loading input data from .mat file at ds-4-2_00",
		"Failure loading output data from .mat file at ds-4-2_00")

def test_load_json():
	ds_loaded = load_data("test/datasets/ds-4-2_00",format="json")
	__check_datasets(ds_loaded,ds_orig,"Failure loading input data from .json file at ds-4-2_00",
		"Failure saving loading data from .json file at ds-4-2_00")

def test_save_json():
	save_data("test/datasets/ds-4-2_00",ds_orig,format="json")
	ds_loaded = load_data("test/datasets/ds-4-2_00.json")
	__check_datasets(ds_loaded,ds_orig,"Failure saving input data to .json file at ds-4-2_00",
		"Failure saving output data to .json file at ds-4-2_00")

def test_save_mat():
	save_data("test/datasets/ds-4-2_01",ds_orig,format="mat")
	ds_loaded = load_data("test/datasets/ds-4-2_01",format="mat")
	__check_datasets(ds_loaded,ds_orig,"Failure saving input data to .mat file at ds-4-2_01",
		"Failure saving output data to .mat file at ds-4-2_01")
	popen("rm -f test/datasets/ds-4-2_01.mat")

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
