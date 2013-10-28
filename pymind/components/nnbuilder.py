from .. import activationfn as af
from ..util import assertType
from .. import errfn as ef
from ..mathutil import create_minimizer

# Deafult setting for Builder
DEFAULT = {
    "input_units" : 1,
    "output_units" : 1,
    "activationfn" : ["identity", "sigmoid"],
    "af" : "sigmoid",
    "bias" : True,
    "errfn" : "squaredError",
    "learn_rate" : 0.7,
    "minimizer": create_minimizer(),
    "iterations" : 10
  }


class Builder(object):
  def __init__(self, setting=None):
    """ Create a builder that can generate iterator of neural network suites.

    A neural network suite consists of:
      
      * Neural Network Settings
        - Layer Units: a list of positive integers denoting number of units for each layer. The
        first and last integers represent the input and output layer respectively, with the rest 
        being the hidden layers.
        The key-value for this is ("layer_units": list of int)
        - Bias: a boolean denoting whether the neural network is biased or not. Default is True.
        The key-value pair for this is ("bias": bool)
        - Activation Functions: a list of string denoting the valid names of the activation
        functions for each layers. See pymind.activationfn for more detail. Default is the sigmoid
        function, except for the input layer, in which case the default is the identity function.
        The key-value pair for this is ("activationfn": list of str) 
      
      * Training Information
        - Input Training Data: a vecotr with each column representing a feature vector for one 
        training example.
        The key-value pair for this is ("X": vector)
        - Output Expected Data: the output vector with each column representing the output vector 
        for one training example.
        The key-value pair for this is ("y": vector)
        - Learning Rate: a number in [0,1] denoting the learning rate for regularization.
        The key-value pair for this is ("learn_rate": float)
        - Error Function: a str denoting the name of the error function used when computing cost. 
        See pymind.errfn for more detail. Default is the squared error function.
        The key-value pair for this is ("errfn": str)
        - Minimizing Function: a minimization function. See pymind.mathutil.create_minimizer for 
        more detail.
        The key-value pair for this is ("minimizer": function)
        - Training Iterations: a positive integer denoting number of times to attempt to minimize 
        the cost with different starting weights.
        The key-value pair for this is ("iterations": int)

    Optional argument:
    setting (dict) - Dicionary denoting the setting for one neural network suite, with one or more 
    of the above key-value pairs. Multiple values (contained in a list) can be provided for one key,
    which signifies multiple suites. For instance, if the learning rates for three suites are 
    2, 3, 4 respectively, the key-value pair would be ("learn_rate": [2,3,4]). Note that to achieve 
    the same thing for layer units and activation functions, the values must be a list of lists of 
    int and a list of lists of str repectively. If multiple values is provided for one setting, the
    number of values provided should match, ie. if two 

    Raises:
    ValueError - if setting contains none of the key-value pairs.
    """
  raise NotImplementedError("Builder constructor not implemented yet.")

def get(self, key, *args):
  """ Retrieve the value for the specified setting, in the form of a dictionary with the setting as
  the key. Multiple settings can be retrieved at once.

  Argument(s):
  key (str) - representing the setting for which the value(s) will be retrieved.

  Returns:
  a dictionary with the specified settings as the dictionary-keys, and their values as the 
  corresponding dictionary-values.

  Raises:
  TypeError
  ValueError
  """
  raise NotImplementedError("Builder.get not implemented yet.")
  

def set(self, **kwargs):
  """ Set the value(s) of the specified setting(s). For example, set(bias=False, iterations=[10,5])
  will set the bias to be False while the number of trainning iterations to be 10 in the first suite
  and 5 in the second suite.

  Arguments:
  key=val - key is a string representing the setting to be changed. val is the value assoicated with
  the setting. See docstring of the constructor for more detail on valid keys and values.

  Return:
  self

  Raises:
  IndexError
  TypeError
  ValueError
  """
  raise NotImplementedError("Builder.set not implemented yet.")

def append(self, **kwargs):
  """ Add new value(s) for the specified setting(s). For example, if the Builder already has its 
  bias set to True and learning rate to be 0.6, then after append(bias=False, learn_rate=[0.1,0.9]),
  the bias will be True in the first suite and False in the second, and the learning rate will be 
  0.6, 0.1, and 0.9 in the first, second, and third suites respectively.

  Arguments:
  key=val - key is a string representing the setting to be appended. val is the value assoicated 
  with the setting. See docstring of the constructor for more detail on valid keys and values.

  Returns:
  self

  Raises:
  TypeError
  ValueError
  """
  raise NotImplementedError("Builder.append not implemented yet.")  

def insert(self, index, **kwargs):
  """ Insert the value(s) of the specified setting at the given index. For example, if the Builder
  already has its learning rate set to 0.5 and 0.9 in the first two suites respectively, then after
  insert(1, learn_rate=0.2), the learning rate will be 0.5, 0.2, and 0.9 in the first three suites
  respectively. Note that unlike Builder.set and Builder.append, this function can only alter one
  setting at a time; error will be thrown if more than one setting is passed in as arguments.

  Arguments:
  index (int) - the index at which the insertion occurs.
  key=val - key is a string representing the setting to be changed. val is the value assoicated 
  with the setting. See docstring of the constructor for more detail on valid keys and values.

  Returns:
  self

  Raises:
  IndexError
  TypeError
  ValueError 
  """
  raise NotImplementedError("Builder.insert not implemented yet.")

def setDefaultActivationFn(self, activationfn):
  """ Change the default activation function used for a layer when none is specififed. Must be a 
  valid activation function name, either represeenting built-in functions or user-defined ones that
  are added to the list of valid functions using pymind.activationfn.add.

  Argument:
  activationfn (str) - representing the activation function the default will be changed to.

  Returns:
  self

  Raises:
  TypeError
  """
  raise NotImplementedError("Builder.setDefaultActivationFn not implemented yet.")

def setDefaultErrorFn(self, errfn):
  """ Change the default error function used for trainning the neural network. Must be a valid error
  function name, either represeenting built-in functions or user-defined ones that are added to the 
  list of valid functions using pymind.errfn.add.

  Argument:
  errfn (str) - representing the error function the default will be changed to.

  Returns:
  self

  Raises:
  TypeError
  """
  raise NotImplementedError("Builder.setDefaultErrorFn not implemented yet.")

def clear(self):
  """ Reset all settings to the default. If the default activation function or error function is
  changed, the new default will be used.

  Returns:
  self
  """
  raise NotImplementedError("Builder.clear not implemented yet.")

def build(self):
  """ Returns an iterator that produces a list of neural network suites according to the settings.

  Returns:
  An iterator that produces a list of neural network suites in the proper order.
  If a setting only has one value, all suites generated will use that value. Otherwise, if multiple
  values are provided for one setting, any other settings with multiple values should have exactly
  two values. That is, the number of values provided should matched.

  Raises:
  ValueError - if the settings cannot be used to construct valid neural networks or to conduct 
  trainning on neural networks.
  """
  raise NotImplementedError("Builder.build not implemented yet.")

def assertPositiveInt(fn, name, var):
  "Helper method for checking if the user input a valid positive integer."
  try:
    var = int(var)
  except (TypeError, ValueError):
    errMsg = "(%s) Expected %s to be integer or something that can be converted to " % (fn, name) \
    + "integer values."
    raise TypeError(errMsg)
  if var <= 0:
    raise ValueError("(%s) %s should be a positive integer." % (fn, name))

def assertLearnRate(fn, name, var):
  "Helper method for checking if the user input a valid learning rate."
  try:
    var = float(var)
  except (TypeError, ValueError):
    errMsg = "(%s) Expected %s to be a number or something that can be converted to " % (fn, name) \
    + "numerical values."
    raise TypeError(errMsg)
  if var < 0 or var > 1.0:
    raise ValueError("The learning rate should be a number between 0 and 1")

