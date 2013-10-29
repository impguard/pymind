from .. import activationfn as af
from ..util import assertType
from .. import errfn as ef
from ..mathutil import create_minimizer
import numpy as np

# Deafult setting for Builder
DEFAULT = {
    "layer_units": [1, 1],
    "af" : "sigmoid",
    "bias" : True,
    "errfn" : "squaredError",
    "learn_rate" : 0.7,
    "minimizer": create_minimizer(),
    "iterations" : 10
  }

# Valid keys for each setting
validKeys = ["layer_units", "activationfn", "bias", "errfn", "learn_rate", "iterations", "X", "y",
  "minimizer"]

checkValues = {}

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
        training example. The size of each column should match the number of input layer units.
        The key-value pair for this is ("X": numpy.ndarray/numpy.matrix)
        - Output Expected Data: the output vector with each column representing the output vector 
        for one training example. The size of each column should match the number of output layer 
        units.
        The key-value pair for this is ("y": numpy.ndarray/numpy.matrix)
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
    int and a list of lists of str repectively. If multiple values is provided for more than one
    setting, the number of values provided should match.

    Raises:
    ValueError - if setting contains none of the key-value pairs.
    TypeError - if the argument is not a dictionary
    """
    self.setting = {}
    if setting == None:
      for key in validKeys:
        if key in DEFAULT:
          self.setting[key] = [ DEFAULT[key] ]
      self.setting["activationfn"] = []
      self.setting["X"] = []
      self.setting['y'] = []
    elif type(setting) == dict:
      fn = "Builder.__init__"
      containsValidKey = False
      for key, values in setting.items():
        if key in validKeys:
          containsValidKey = True
          self.setting[key] = checkValues[key](fn, values)
        else:
          raise ValueError("(%s) Unknown setting: %s" % (fn, key))
      # Raises error if setting contains none of the valid keys.
      if not containsValidKey:
        errMsg = "(%s) Expected %s to contains at least one valid key-value pair." % (fn, "setting")
        raise ValueError(errMsg)
      # Every setting not set by user should be default
      for key in validKeys:
        if key not in setting and key in DEFAULT:
          self.setting[key] = [ DEFAULT[key] ]
      if "activationfn" not in setting:
        self.setting["activationfn"] = []
      if "X" not in setting:
        self.setting["X"] = []
      if "y" not in setting:
        self.setting["y"] = []
    else:
      raise TypeError("Argument for Builder constructor must be a dictionary.")

  def get(self, key, *args):
    """ Retrieve the value for the specified setting. Multiple settings can be retrieved at once, a 
    list will be returned in this case.

    Argument(s):
    key (str) - representing the setting for which the value(s) will be retrieved.

    Returns:
    value or a list of values. The type of the value varies depending on the corresponding setting.

    Raises:
    TypeError
    ValueError
    """
    fn = "Builder.get"
    assertValidKey(fn, key)
    if len(args) == 0:
      data = self.setting[key]
      if len(data) == 1:
        values = data[0]
      elif len(data) > 1:
        values = data
      else:
        values = []
    else:
      values = []
      data = self.setting[key]
      if len(data) == 1:
        values.append(data[0])
      elif len(data) > 1:
        values.append(data)
      else:
        values.append([])
      for k in args:
        assertValidKey(fn, k)
        data = self.setting[k]
        if len(data) == 1:
          values.append(data[0])
        elif len(data) > 1:
          values.append(data)
        else:
          values.append([])
    return values

  def getSetting(self):
    """ Retrieve the value for the every settings, in the form of a dictionary with the setting as
    the key.

    Returns:
    a dictionary with the settings as the dictionary-keys, and their values as the corresponding 
    dictionary-values.
    """
    info = {}
    for k, v in self.setting.items():
      if len(v) == 1:
        info[k] = v[0]
      elif len(v) > 1:
        info[k] = v
      else:
        info[k] = []
    return info


  def set(self, **kwargs):
    """ Set the value(s) of the specified setting(s). For example, set(bias=False,iterations=[10,5])
    will set the bias to be False while the number of trainning iterations to be 10 in the first 
    suite and 5 in the second suite.

    Arguments:
    key=val - key is a string representing the setting to be changed. val is the value assoicated 
    with the setting. See docstring of the constructor for more detail on valid keys and values.

    Return:
    self

    Raises:
    TypeError
    ValueError
    SyntaxError
    """
    fn = "Builder.set"
    if len(kwargs) == 0:
      raise TypeError("(%s) Expected at least 1 argument, got 0" % fn)
    containsValidKey = False
    for key, values in kwargs.iteritems():
      if key in validKeys:
        containsValidKey = True
        self.setting[key] = checkValues[key](fn, values)
      else:
        raise ValueError("(%s) Unknown setting: %s" % (fn, key))
    if not containsValidKey:
      errMsg = "(%s) Expected at least one valid key-value pair." % fn
      raise ValueError(errMsg)
    return self

  def append(self, **kwargs):
    """ Add new value(s) for the specified setting(s). For example, if the Builder already has its 
    bias set to True and learn rate to be 0.6, then after append(bias=False, learn_rate=[0.1,0.9]),
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
    SyntaxError
    """
    fn = "Builder.append"
    if len(kwargs) == 0:
      raise TypeError("(%s) Expected at least 1 argument, got 0" % fn)
    containsValidKey = False
    for key, values in kwargs.iteritems():
      if key in validKeys:
        containsValidKey = True
        self.setting[key].extend(checkValues[key](fn, values))
      else:
        raise ValueError("(%s) Unknown setting: %s" % (fn, key))
    if not containsValidKey:
      errMsg = "(%s) Expected at least one valid key-value pair." % fn
      raise ValueError(errMsg)
    return self
    

  def insert(self, index, **kwargs):
    """ Insert the value(s) of the specified setting at the given index. For example, if the Builder
    already has its learn rate set to 0.5 and 0.9 in the first two suites respectively, then after
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
    SyntaxError
    """
    raise NotImplementedError("Builder.insert not implemented yet.")

  def setDefaultActivationFn(self, activationfn):
    """ Change the default activation function that is used for a layer when none is specififed. Must 
    be a valid activation function name, either represeenting built-in functions or user-defined ones 
    that are added to the list of valid functions using pymind.activationfn.add.

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

  def remove(self, key, index=None):
    """ Remove a value of the setting, specified by key, at the given index. If no index is specified,
    remove teh last value of the setting specified by key. Use Builder.clear to remove every value of 
    a setting.

    Arguments:
    key (str) - representing the setting to be changed.
    index (int) - the index at which the value is removed.

    Returns:
    the value removed

    Raises:
    IndexError
    """
    raise NotImplementedError("Builder.remove not implemented yet.")
    

  def clear(self, *args):
    """ Reset the specified setting(s) to the default. If the default activation function or error 
    function is changed, the new default will be used. If no setting is specified, all settings will 
    be reset to the defautl.

    Arguments:
    key (str) - representing the setting to be cleared.

    Returns:
    self
    """
    raise NotImplementedError("Builder.clear not implemented yet.")

  def build(self):
    """ Returns an iterator that produces a list of neural network suites according to the settings.

    If a setting only has one value, all suites generated will use that value. Otherwise, if multiple
    values are provided for one setting, any other settings with multiple values should have exactly
    two values. That is, the number of values provided should matched.

    All settings except input (X) and output (y) trainning data can be left unset. Either the user 
    never give the setting a value or the value of the is removed (ie. Builder.get returns empty list 
    for the setting). In both cases the default will be used. An exception being the activation 
    functions, which if left unset, its value will be infered from the layer units (using identity
    function for input layer, and default activation function for other layers). Note that the number
    of activation functions and number of layers must match, else an error will be thrown.

    Returns:
    An iterator that produces a list of neural network suites in the proper order.
    Each suite is a dictionary with the following key-value pairs:
      ("layer_units": list of int)
      ("activationfn": list of str)
      ("bias": bool)
      ("learn_rate": float)
      ("errfn": str)
      ("minimizer": function)
      ("iterations": int)
      ("X": numpy.ndarray)
      ("y": numpy.ndarray)

    Raises:
    ValueError - if the settings cannot be used to construct valid neural networks or to conduct 
    trainning on neural networks.
    """
    raise NotImplementedError("Builder.build not implemented yet.")

def assertPositiveInt(fn, name, var, layer_units=True):
  """ Helper method for checking if the user input a valid positive integer."""
  try:
    var = int(var)
  except (TypeError, ValueError):
    if not layer_units:
      errMsg = "(%s) Expected %s to be integer or something that can be converted " % (fn, name)\
      + "to integer value."
    else:
      errMsg = "(%s) Expected %s to be a list of integers or something that can be " % (fn, name)\
      + "converted to a list of integers."
    raise TypeError(errMsg)
  if var <= 0:
    if not layer_units:
      raise ValueError("(%s) %s should be a positive integer." % (fn, name))
    else:
      raise ValueError("(%s) %s should be a list of positive integers." % (fn, name))

def assertLearnRate(fn, name, var):
  """ Helper method for checking if the user input a valid learning rate."""
  try:
    var = float(var)
  except (TypeError, ValueError):
    errMsg = "(%s) Expected %s to be a number or something that can be converted " % (fn, name) \
    + "to numerical values."
    raise TypeError(errMsg)
  if var < 0 or var > 1.0:
    raise ValueError("The learning rate should be a number between 0 and 1")

def assertFn(fn, name, var, activationfn=True):
  """ Helper method for checking if the user input is a valid activation or error function name."""
  assertType(fn, name, var, str)
  if (not af.contains(var)) and (not ef.contains(var)):
    if activationfn:
      errMsg = "%s is not a valid activation function name." % var
    else:
      errMsg = "%s is not a valid error function name." % var
    raise ValueError(errMsg)

def assertTrainningData(fn, name, var, input=True):
  """ Helper method for checking if the user input is valid input/output data."""
  if (type(var) is not np.ndarray) and (type(var) is not np.matrix):
    dataType = "input trainning data" if input else "expected output data"
    raise TypeError("(%s) Expected %s (%s) to be numpy array or matrix." % (fn, name, dataType))

def assertValidKey(fn, key):
  """ Helper method for checking if the key corresponds to a setting."""
  if key not in validKeys:
    raise TypeError("(%s) %s doesn't correspond to any setting." % (fn, key))

def checkLayerUnits(fn, values):
  """ Helper method for making sure the values are the correct format for the layer units."""
  if hasattr(values, "__len__") and len(values) > 0:
    newValues = []
    # Multiple values for layer units, ie. list of lists of int
    if hasattr(values[0], "__len__"):
      for value in values:
        temp = []
        for unit in value:
          assertPositiveInt(fn, "layer_units", unit)
          temp.append(int(unit))
        newValues.append(temp)
    # One value for layer units, ie. list of int
    else:
      temp = []
      for unit in values:
        assertPositiveInt(fn, "layer_units", unit)
        temp.append(int(unit))
      newValues.append(temp)
  else:
    raise TypeError("(%s) Expected layer_units to be a list of int or a list of list of int." % fn)
  return newValues

checkValues["layer_units"] = checkLayerUnits

def checkActivationFn(fn, values):
  """ Helper method for making sure the activation function names are valid."""
  if (type(values) == list or type(values) == tuple) and len(values) > 0:
    newValues = []
    # Multiple values for activation functions, ie. list of lists of str
    if (type(values[0]) == list or type(values[0]) == tuple) and len(values[0]) > 0:
      for value in values:
        temp = []
        for name in value:
          assertFn(fn, "activationfn", name)
          temp.append(name)
        newValues.append(temp)
    # One value for activation functions, ie. list of str
    else:
      temp = []
      for name in values:
        assertFn(fn, "activationfn", name)
        temp.append(name)
      newValues.append(temp)
  else:
    raise TypeError("(%s) Expected activationfn to be a list of str or a list of list of str." % fn)
  return newValues

checkValues["activationfn"] = checkActivationFn

def checkBias(fn, values):
  """ Helper method for checking if bias is valid."""
  newValues = []
  if hasattr(values, "__len__") and len(values) > 0:
    for value in values:
      assertType(fn, "bias", value, bool)
      newValues.append(value)
  else:
    assertType(fn, "bias", values, bool)
    newValues.append(values)
  return newValues

checkValues["bias"] = checkBias

def checkErrFn(fn, values):
  """ Helper method for checking if error function is valid."""
  newValues = []
  if (type(values) == list or type(values) == tuple) and len(values) > 0:
    for value in values:
      assertFn(fn, "errfn", value, False)
      newValues.append(value)
  else:
    assertFn(fn, "errfn", values, False)
    newValues.append(values)
  return newValues

checkValues["errfn"] = checkErrFn

def checkLearnRate(fn, values):
  """ Helper method for checking if learning rate is valid."""
  newValues = []
  if hasattr(values, "__len__") and len(values) > 0:
    for value in values:
      assertLearnRate(fn, "learn_rate", value)
      newValues.append(value)
  else:
    assertLearnRate(fn, "learn_rate", values)
    newValues.append(values)
  return newValues

checkValues["learn_rate"] = checkLearnRate

def checkIterations(fn, values):
  """ Helper method for checking if number of iterations is valid."""
  newValues = []
  if hasattr(values, "__len__") and len(values) > 0:
    for value in values:
      assertPositiveInt(fn, "iterations", value, False)
      newValues.append(value)
  else:
    assertPositiveInt(fn, "iterations", values, False)
    newValues.append(values)
  return newValues

checkValues["iterations"] = checkIterations

def checkData(fn, values, input):
  """ Helper method for checking if input/output trainning data is valid."""
  newValues = []
  if (type(values) == list or type(values) == tuple) and len(values) > 0:
    for value in values:
      assertTrainningData(fn, "X" if input else "y", value, input)
      newValues.append(value)
  else:
    assertTrainningData(fn, "X" if input else "y", values, input)
    newValues.append(values)
  return newValues

def checkX(fn, values):
  return checkData(fn, values, True)

checkValues["X"] = checkX

def checkY(fn, values):
  return checkData(fn, values, False)

checkValues["y"] = checkY


def checkMinimizer(fn, values):
  """ Helper method for minimizer."""
  newValues = []
  if hasattr(values, "__len__") and len(values) > 0:
    newValues = values
  else:
    if values == None or values == []:
      raise TypeError("(%s) Expected %s to be a function." % (fn, name))
    newValues.append(values)
  return newValues

checkValues["minimizer"] = checkMinimizer