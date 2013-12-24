import numpy as np
import json
import scipy.io
from pymind.activationfn import *
from pymind.components import *

load_routines = {}
save_routines = {}

"""
Builder class for training data. Used to construct a dataset from scratch.
"""
class DatasetBuilder(object):

  def __init__(self, icount, ocount):
    """  Constructs a new Datasetbuilder.

    Parameters:
      icount, the number of inputs to the neural network
      ocount, the number of outputs from the neural network
    """
    self.X = [list() for _ in xrange(icount)]
    self.y = [list() for _ in xrange(ocount)]
    self.icount = icount
    self.ocount = ocount

  def add(self, ivec, ovec):
    """  Adds a datapoint to this DatasetBuilder.

    Parameters:
      ivec, a vector (list or array) of input features. Must be the same length as self.icount
      ovec, a vector (list or array) of output values. Must be the same length as self.ocount
    """
    assert len(ivec) == self.icount, "Vector does not match input data."
    assert len(ovec) == self.ocount, "Vector does not match output data."
    for k, data in enumerate(ivec):
      self.X[k].append(data)
    for k, data in enumerate(ovec):
      self.y[k].append(data)

  def build(self):
    """ Returns a dictionary containing matrices X and y, consisting of the training data added to
    DatasetBuilder.  X is mapped to an xa by xb array, where xa is the  number of inputs and xb is
    the number of training samples.  y is mapped to an ya by yb array, where ya is the number of
    outputs and yb is the number of training samples.
    """
    return {"X":np.matrix(self.X), "y":np.matrix(self.y)}

def save_data(fname, data, format=None):
  """ Given a file name "fname", format "format" and a dataset "data", attempts to save data to file
  formatted using "format" such that it can be loaded using load_data. If format is not specified,
  attempts to search the file name for an extension.

  Parameters:
    fname, the name of the target file
    data, the dataset to save
    format, the format of the output file
  """
  if format is None:
    dot = fname.rfind(".")
    if dot != -1:
      format = fname[dot+1:]
    else:
      raise RuntimeError("Please specify a format for file " + fname)
  elif len(format) > 0 and format[0]==".":
    format = format[1:]
  if format in load_routines:
    return save_routines[format](fname, data)
  else:
    raise RuntimeError("Unrecognized file format \"" + "." + format + "\"")

def __save_json_data(fname, data):
  """ Given a file name "fname" and a dataset "data", saves data to <fname>.json such that it can be
  loaded using load_data or __load_json_data.

  Parameters:
    fname, the name of the target file
    data, the dataset to save
  """
  if ".json" != fname[-5:]:
    fname = fname + ".json"
  fout = open(fname, "w")
  out = {"X":[], "y":[]}
  for x in data["X"]:
    d = []
    for i in xrange(x.shape[1]):
      d.append(float(x[0, i]))
    out["X"].append(d)
  for y in data["y"]:
    d = []
    for i in xrange(y.shape[1]):
      d.append(float(y[0, i]))
    out["y"].append(d)
  enc = json.JSONEncoder()
  out = enc.encode(out)
  fout.write(out)
  fout.close()
save_routines["json"] = __save_json_data

def __save_mat_data(fname, data):
  """ Given a file name "fname" and a dataset "data", saves data to <fname>.mat such that it can be
  loaded using load_data or __load_mat_data.

  Parameters:
    fname, the name of the target file
    data, the dataset to save
  """
  if ".mat" != fname[-5:]:
    fname = fname + ".mat"
  scipy.io.savemat(fname, data, oned_as="row")
save_routines["mat"] = __save_mat_data

def load_data(fname, format=None):
  """ Given a file name "fname" and a string "format" indicating the file format, attempts to load
  and return the training data contained within the file. If no format is specified, attempts to
  search the file name for an extension.

  Parameters:
    fname, the name of a file containing a training dataset
    format, the format of the input file
  """
  if format is None:
    dot = fname.rfind(".")
    if dot != -1:
      format = fname[dot+1:]
    else:
      raise RuntimeError("Please specify a format for file " + fname)
  elif len(format) > 0 and format[0]==".":
    format = format[1:]
  if format in load_routines:
    return load_routines[format](fname)
  else:
    raise RuntimeError("Unrecognized file format \"" + "." + format + "\"")

def __load_json_data(fname):
  """ Converts a JSON training dataset into Numpy matrix format.

  Parameters:
    fname, the name of a JSON file consisting of 2 keys: "X" which binds to an array of arrays
      representing the list of input vectors  and "y" which binds to an array of arrays representing
      the list of output vectors.
  """
  if ".json" != fname[-5:]:
    fname = fname + ".json"
  jsfile = open(fname)
  ds = json.load(jsfile)
  jsfile.close()
  X, y = np.matrix(ds[u"X"]), np.matrix(ds[u"y"])
  return {"X":X, "y":y}
load_routines["json"] = __load_json_data

def __load_mat_data(fname):
  """ Converts a matlab training dataset into Numpy matrix format.

  Parameters:
    fname, the name of a matlab file consisting of 2 keys: "X" which binds to an array of arrays
    representing the list of input vectors  and "y" which binds to an array of arrays representing
    the list of output vectors.
  """
  ds = scipy.io.loadmat(fname)
  X, y = np.matrix(ds["X"]), np.matrix(ds["y"])
  return {"X":X, "y":y}
load_routines["mat"] = __load_mat_data

def split_data(X, y=None, parts=2):
  """ Randomly partitions a set of training data into multiple parts

  Parameters:
    X, a matrix representing the inputs for the training data. Alternately, could be a dictionary
      containing both "X" and "y" as keys mapped to matrices
    y, a matrix representing the outputs for the training data
    parts, the number of parts into which the training data will be split, or a list indicating the
      proportions of each part into which we split the data
  """
  if y is None and type(X) is dict:
    y = X["y"]
    X = X["X"]
  if hasattr(parts, "__len__"):
    kparts = reduce(lambda x, y:x+y, parts)
    dsparts, dsets = split_data(X, y , kparts), []
    for part in parts:
      head, dsparts = dsparts[:part], dsparts[part:]
      dsets.append({"X":np.hstack([head[i]["X"] for i in xrange(part)]),
        "y":np.hstack([head[i]["y"] for i in xrange(part)])})
    return dsets
  else:
    scount = int(X.shape[1])
    assert scount==y.shape[1], "Invalid dataset, number of inputs must match number of outputs"
    a = np.arange(scount)
    np.random.shuffle(a)
    start, inc = 0.0, scount/parts
    end, dsets = inc, []
    for _ in xrange(parts):
      indices = a[round(start):round(end)]
      dsets.append({"X":X[:, indices], "y":y[:, indices]})
      start = end
      end += inc
    return dsets

def __matrixToList(mtx):
  """ Converts a numpy matrix into a 2D Python list. """
  arr = []
  for row in mtx:
    arr.append([t[1] for t in np.ndenumerate(row)])
  return arr

def save_neural_net(fname, nnet, format="json"):
  """ Given a file name, neural network and a format serializes a neural network into the specified
  format. File contains the following information: the size of each hidden layer, number of input
  units, number of output units, each layer"s activation function, whether or not the network is
  biased, and the weight of each link in the network.

  Parameters:
    fname, the name of the file (may include an extension)
    nnet, the neural network to serialize
    format, the file format to use
  """
  if format == "json" or ".json" == fname[-5:]:
    __save_json_neural_net(fname, nnet)
  elif format == "mat" or ".mat" == fname[-4:]:
    __save_mat_neural_net(fname, nnet)

def __save_json_neural_net(fname, nnet):
  """ Given a file name and neural network, serializes the neural network as a json file. See doc
  for save_neural_net for more information.

  Parameters:
    fname, the name of the file
    nnet, the neural network to serialize
  """
  obj = {}
  obj["hidden_units"] = nnet.hidden_units
  obj["input_units"] = nnet.input_units
  obj["output_units"] = nnet.output_units
  obj["bias"] = nnet.bias
  aflist = []
  for af in nnet.activationfn:
    if af is sigmoid:
      aflist.append("sigmoid")
    elif af is identity:
      aflist.append("identity")
    else:
      aflist.append("unknown")
  obj["activationfn"] = aflist
  obj["weights"] = [__matrixToList(t) for t in nnet.weights]
  enc = json.JSONEncoder()
  out = enc.encode(obj)
  if ".json" not in fname[-5:]:
    fname = fname + ".json"
  fout = open(fname, "w")
  fout.write(out)
  fout.close()

def __save_mat_neural_net(fname, nnet):
  """ Given a file name and neural network, serializes the neural network as a mat file. See doc
  for save_neural_net for more information.

  Parameters:
    fname, the name of the file
    nnet, the neural network to serialize
  """
  raise NotImplementedError("Saving neural networks to .mat files is not yet supported.")

def load_neural_net(fname, format="json"):
  """ Given a file name "fname" and a string "format" indicating the file format, attempts to load
  and return the neural network contained within the file. If no format is specified, attempts to
  search the file name for an extension.

  Parameters:
    fname, the name of a file containing a training dataset
    format, the format of the input file
  """
  if format == "json" or ".json" == fname[-5:]:
    return __load_json_neural_net(fname)
  elif format == "mat" or ".mat" == fname[-4:]:
    return __load_mat_neural_net(fname)

def __load_json_neural_net(fname):
  """ Given a file name, deserializes the neural network as a json file. See doc for load_neural_net
  for more information.

  Pameters:
    fname, the name of the file
  """
  if ".json" not in fname[-5:]:
    fname = fname + ".json"
  fin = open(fname)
  rawstr = fin.read()
  dec = json.JSONDecoder()
  obj = dec.decode(rawstr)
  params = {}
  params["hidden_units"] = obj["hidden_units"]
  params["input_units"] = obj["input_units"]
  params["output_units"] = obj["output_units"]
  params["bias"] = obj["bias"]
  try:
    # type of each afname in obj is unicode, not str
    params["activationfn"] = [get(str(afname)) for afname in obj["activationfn"]]
  except AssertionError:
    raise RuntimeError("Error: Loading custom activation functions is not yet supported.")
  nnet = NeuralNetwork(params)
  nnet.weights = [np.matrix(t) for t in obj["weights"]]
  return nnet

def __load_mat_neural_net(fname):
  """ Given a file name, deserializes the neural network as a mat file. See doc for load_neural_net
  for more information.

  Parameters:
    fname, the name of the file
  """
  raise NotImplementedError("Loading neural networks from .mat files is not yet supported.")
