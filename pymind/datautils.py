import numpy as np
import json
import scipy.io

load_routines = {}

"""
Builder class for training data. Used to construct a dataset from scratch.
"""
class DatasetBuilder(object):

  """  Constructs a new Datasetbuilder.

  Parameters:
    icount, the number of inputs to the neural network
    ocount, the number of outputs from the neural network
  """
  def __init__(self,icount,ocount):
    self.X = [list() for _ in xrange(icount)]
    self.y = [list() for _ in xrange(ocount)]
    self.icount = icount
    self.ocount = ocount

  """  Adds a datapoint to this DatasetBuilder.

  Parameters:
    ivec, a vector (list or array) of input features. Must be the same length as self.icount
    ovec, a vector (list or array) of output values. Must be the same length as self.ocount
  """
  def add(self,ivec,ovec):
    assert len(ivec) == self.icount, 'Vector does not match input data.'
    assert len(ovec) == self.ocount, 'Vector does not match output data.'
    for k,data in enumerate(ivec):
      self.X[k].append(data)
    for k,data in enumerate(ovec):
      self.y[k].append(data)

  """ Returns a dictionary containing matrices X and y, consisting of the training data added to
  DatasetBuilder.  X is mapped to an xa by xb array, where xa is the  number of inputs and xb is the
  number of training samples.  y is mapped to an ya by yb array, where ya is the number of outputs
  and yb is the number of training samples.
  """
  def build(self):
    return {'X':np.matrix(self.X),'y':np.matrix(self.y)}

""" Given a file name 'fname' and a string 'format' indicating the file format, attempts to load and
return the training data contained within the file. If no format is specified, attempts to search
the file name for an extension.

Parameters:
  fname, the name of a file containing a training dataset
  format, the format of the input file
"""
def load_data(fname,format=None):
  if format is None:
    dot = fname.rfind('.')
    if dot != -1:
      format = fname[dot+1:]
    else:
      raise RuntimeError('Please specify a format for file ' + fname)
  elif len(format) > 0 and format[0]=='.':
    format = format[1:]
  if format in load_types:
    return load_routines[format](fname)
  else:
    raise RuntimeError('Unrecognized file format \"' + '.' + format + '\"')

""" Converts a JSON training dataset into Numpy matrix format.

Parameters:
  fname, the name of a JSON file consisting of 2 keys: 'X' which binds to an array of arrays
    representing the list of input vectors  and 'y' which binds to an array of arrays representing
    the list of output vectors.
"""
def __load_json_data(fname):
  if '.json' != fname[-5:]:
    fname = fname + '.json'
  jsfile = open(fname)
  ds = json.load(jsfile)
  jsfile.close()
  X,y = np.matrix(ds[u'X']),np.matrix(ds[u'y'])
  return {'X':X,'y':y}
load_routines['json'] = __load_json_data

""" Converts a matlab training dataset into Numpy matrix format.

Parameters:
  fname, the name of a matlab file consisting of 2 keys: 'X' which binds to an array of arrays
  representing the list of input vectors  and 'y' which binds to an array of arrays representing the
  list of output vectors.
"""
def __load_mat_data(fname):
  ds = scipy.io.loadmat(fname)
  X,y = np.matrix(ds['X']),np.matrix(ds['y'])
  return {'X':X,'y':y}
load_routines['mat'] = __load_mat_data

""" Randomly partitions a set of training data into multiple parts

Parameters:
  X, a matrix representing the inputs for the training data
  y, a matrix representing the outputs for the training data
  parts, the number of parts into which the training data will be split
"""
def split_data(X,y,parts=2):
  scount = int(X.shape[1])
  assert scount==y.shape[1], 'Invalid dataset, number of inputs must match number of outputs'
  a = np.arange(scount)
  np.random.shuffle(a)
  start,inc = 0.0,scount/parts
  end,dsets = inc,[]
  for _ in xrange(parts):
    indices = a[round(start):round(end)]
    dsets.append({'X':X[:,indices],'y':y[:,indices]})
    start = end
    end += inc
  return dsets
