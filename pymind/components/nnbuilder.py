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

# Valid keys that is used to access and change the corresponding settings.
validKeys = ["layer_units", "activationfn", "bias", "errfn", "learn_rate", "iterations", "X", "y",
  "minimizer"]

# Functions for making sure the user-input values for a setting is the correct format.
checkValues = {}

class Builder(object):
  """ Create a builder that can generate an iteratior of neural network suites.

    A neural network suite consists of:
      
      * Neural Network Settings
        - Layer Units ("layer_units": list of int): a list of positive integers denoting number of 
            units for each layer. The first and last integers represent the input and output layer 
            respectively, with the rest being the hidden layers.
            The default value is [1, 1], denoting one input and one output layer unit respectively.
        - Bias ("bias": bool): a boolean denoting whether the neural network is biased or not.
            The default value is True.
        - Activation Functions ("activationfn": list of str): a list of string denoting the valid 
            names of the activation functions for each layers. 
            See pymind.activationfn for more detail. 
            There is no default value, if no value is input for this setting, the activation 
            functions will be inferred from the number of layers, with identity function for the 
            input layer, and the default activation function for the other layers. The default 
            activation function can be accessed via Builder.getDefaultActivationFn and changed via 
            Builder.setDefaultActivationFn
      
      * Training Information
        - Input Training Data ("X": numpy.ndarray OR numpy.matrix): a vector with each column 
            representing a feature vector for one training example. The size of each column should 
            match the number of input layer units.
            There is no default value for this setting. This setting must have at least one value 
            before neural network suite(s) can be generated.
        - Output Expected Data ("y": numpy.ndarray OR numpy.matrix): a vector with each column
            representing the output vector for one training example. The size of each column should 
            match the number of output layer units.
            There is no default value for this setting. This setting must have at least one value 
            before neural network suite(s) can be generated.
        - Learning Rate ("learn_rate": float): a number in [0,1] denoting the learning rate for 
            regularization.
            The default is 0.7
        - Error Function ("errfn": str): a str denoting the name of the error function used when 
            computing cost. See pymind.errfn for more detail. 
            The default value is "squaredError", representing the squared error function.
        - Minimizing Function ("minimizer": function): a minimization function. 
            See pymind.mathutil.create_minimizer for more detail.
        - Training Iterations ("iterations": int): a positive integer denoting number of times to 
            attempt to minimize the cost with different starting weights.
            The default value is 10.
  """
  def __init__(self, setting=None):
    """ Create a builder using the default settings or via user settings by passing in a dictionary
    of parameters. 

    The parameters that can be passed in includes:
      Layer Units ("layer_units": list of int)
      Bias ("bias": bool)
      Activation Functions ("activationfn": list of str)
      Input Training Data ("X": numpy.ndarray OR numpy.matrix)
      Output Expected Data ("y": numpy.ndarray OR numpy.matrix)
      Learning Rate ("learn_rate": float)
      Error Function ("errfn": str)
      Minimizing Function ("minimizer": function)
      Training Iterations ("iterations": int)
    Multiple values (contained in a list) can be provided for one key, which signifies multiple 
    suites. For instance, if the learning rates for three suites are 2, 3, 4 respectively, the 
    key-value pair would be ("learn_rate": [2,3,4]). Note that to achieve the same behavior for 
    layer units and activation functions, the values must be a list of lists of int and a list of 
    lists of str respectively.

    Raises:
    ValueError - if setting contains none of the key-value pairs.
    TypeError - if the argument is not a dictionary
    """
    self.settings = {}
    self.defaultaf = DEFAULT['af']
    if setting == None:
      for key in validKeys:
        if key in DEFAULT:
          self.settings[key] = [ DEFAULT[key] ]
      self.settings["activationfn"] = []
      self.settings["X"] = []
      self.settings['y'] = []
    elif type(setting) == dict:
      fn = "Builder.__init__"
      # Raises error if setting contains none of the valid keys.
      if len(setting) == 0:
        errMsg = "(%s) Expected %s to contains at least one valid key-value pair." % (fn, "setting")
        raise ValueError(errMsg)
      for key, values in setting.items():
        _assertValidKey(fn, key)
        containsValidKey = True
        self.settings[key] = checkValues[key](fn, values)
      # Every setting not set by user should be default
      for key in validKeys:
        if key not in setting and key in DEFAULT:
          self.settings[key] = [ DEFAULT[key] ]
      if "activationfn" not in setting:
        self.settings["activationfn"] = []
      if "X" not in setting:
        self.settings["X"] = []
      if "y" not in setting:
        self.settings["y"] = []
    else:
      raise TypeError("Argument for Builder constructor must be a dictionary.")

  def get(self, key, *args):
    """ Retrieve the value for the specified setting. Multiple settings can be retrieved at once, a 
    list will be returned in this case.

    Argument(s):
    key (str) -- representing the setting for which the value(s) will be retrieved.

    Returns:
    A value or a list of values. The type of the value varies depending on the corresponding 
      setting.

    Raises:
    TypeError
    ValueError
    """
    fn = "Builder.get"
    _assertValidKey(fn, key)
    if len(args) == 0:
      data = self.settings[key]
      if len(data) == 1:
        values = data[0]
      elif len(data) > 1:
        values = data
      else:
        values = []
    else:
      values = []
      data = self.settings[key]
      if len(data) == 1:
        values.append(data[0])
      elif len(data) > 1:
        values.append(data)
      else:
        values.append([])
      for k in args:
        _assertValidKey(fn, k)
        data = self.settings[k]
        if len(data) == 1:
          values.append(data[0])
        elif len(data) > 1:
          values.append(data)
        else:
          values.append([])
    return values

  def getSetting(self):
    """ Retrieve the value for every settings, in the form of a dictionary with the setting as the 
    key.

    Returns:
    A dictionary with the settings as the dictionary-keys, and their values as the corresponding 
      dictionary-values.
    """
    info = {}
    for k, v in self.settings.items():
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
    key=val -- key is a string representing the setting to be changed. val is the value assoicated 
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
    for key, values in kwargs.iteritems():
      _assertValidKey(fn, key)
      containsValidKey = True
      self.settings[key] = checkValues[key](fn, values)
    return self

  def append(self, **kwargs):
    """ Add new value(s) for the specified setting(s). For example, if the Builder already has its 
    bias set to True and learn rate to be 0.6, then after append(bias=False, learn_rate=[0.1,0.9]),
    the bias will be [True, False], and the learning rate will be [0.6, 0.1, 0.9].

    Arguments:
    key=val -- key is a string representing the setting to be appended. val is the value assoicated 
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
    for key, values in kwargs.iteritems():
      _assertValidKey(fn, key)
      containsValidKey = True
      self.settings[key].extend(checkValues[key](fn, values))
    return self
    

  def insert(self, index, **kwargs):
    """ Insert the value(s) of the specified setting at the given index. For example, if the Builder
    already has its learn rate set to 0.5 and 0.9 in the first two suites respectively, then after
    insert(1, learn_rate=0.2), the learning rate will be 0.5, 0.2, and 0.9 in the first three suites
    respectively. Note that unlike Builder.set and Builder.append, this function can only alter one
    setting at a time; error will be thrown if more than one setting is passed in as arguments.

    Arguments:
    index (int) -- the index at which the insertion occurs.
    key=val -- key is a string representing the setting to be changed. val is the value assoicated 
      with the setting. See docstring of the constructor for more detail on valid keys and values.

    Returns:
    self

    Raises:
    IndexError
    TypeError
    ValueError 
    SyntaxError
    """
    fn = "Builder.append"
    assertType(fn, "index", index, int)
    if len(kwargs) == 0:
      raise TypeError("(%s) Takes in exactly 2 arguments, got 1" % fn)
    elif len(kwargs) > 1:
      raise TypeError("(%s) Takes in exactly 2 arguments" % fn)
    else:
      for key, values in kwargs.iteritems():
        if key in validKeys:
          if (type(values) == list or type(values) == tuple):
            self.settings[key][index:index] = values
          else:
            self.settings[key].insert(index, values)
        else:
          raise ValueError("(%s) Unknown setting: %s" % (fn, key))
    return self

  def setDefaultActivationFn(self, activationfn):
    """ Change the default activation function that is used for a layer when none is specififed.
    Must be a valid activation function name, either represeenting built-in functions or 
    user-defined ones that are added to the list of valid functions using pymind.activationfn.add.

    Argument:
    activationfn (str) -- representing the activation function the default will be changed to.

    Returns:
    self

    Raises:
    TypeError
    """
    fn = "Builder.setDefaultActivationFn"
    _assertFn(fn, "activationfn", activationfn)
    self.defaultaf = activationfn
    return self

  def getDefaultActivationFn(self):
    """ Returns the default activation function.

    Returns:
    The default activation function
    """
    return self.defaultaf

  def remove(self, key, index=None):
    """ Remove a value of the setting, specified by key, at the given index. If no index is 
    specified, remove the last value of the setting specified by key. Use Builder.clear to remove 
    every value of a setting.

    Arguments:
    key (str) -- representing the setting to be changed.
    index (int) -- the index at which the value is removed.

    Returns:
    The value removed

    Raises:
    IndexError
    """
    fn = "Builder.remove"
    _assertValidKey(fn, key)
    if index == None:
      removed = self.settings[key].pop()
    else:
      removed = self.settings[key].pop(index)
    return removed 

  def clear(self, *args):
    """ Reset the specified setting(s) to the default. If the default activation function or error 
    function is changed, the new default will be used. If no setting is specified, all settings will 
    be reset to the defautl.

    Arguments:
    key (str) -- representing the setting to be cleared.

    Returns:
    self
    """
    fn = "Builder.clear"
    if len(args) == 0:
      for key in validKeys:
        if key in DEFAULT:
          self.settings[key] = [ DEFAULT[key] ]
      self.settings["activationfn"] = []
      self.settings["X"] = []
      self.settings['y'] = []
    else:
      for key in args:
        _assertValidKey(fn, key)
        if key in DEFAULT:
          self.settings[key] = [ DEFAULT[key] ]
        else:
          self.settings[key] = []
    return self

  def build(self):
    """ Returns an iterator that produces a list of neural network suites according to the settings.

    If a setting only has one value, all suites generated will use that value. Otherwise, if 
    multiple values are provided for one setting, any other settings with multiple values should 
    have exactly two values. That is, the number of values provided should matched.

    All settings except input (X) and output (y) trainning data can be left unset. Either the user 
    never give the setting a value or the value of the is removed (ie. Builder.get returns empty 
    list for the setting). In both cases the default will be used. An exception being the activation 
    functions, which if left unset, its value will be infered from the layer units (using identity 
    function for input layer, and default activation function for other layers). 

    Note that following must match, else an error will be thrown: 
      number of activation functions and the number of layers
      number of input data in one trainning example (size of a column in X) and the number of input 
        layer units (first entry in the list of int)
      number of output data in one trainning example (size of a column in y) and the number of 
        output layer units (last entry in the list of int)

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
    fn = "Builder.build"
    numSuites = 1
    for key, values in self.settings.items():
      if len(values) > numSuites:
        numSuites = len(values)
    _assertValidSetting(fn, self.settings, numSuites)

    return self._suiteIterator(self.settings, self.defaultaf, numSuites)

  class _suiteIterator(object):
    def __init__(self, setting, defaultaf, numSuites):
      self.settings = dict(setting)
      self.numSuites = numSuites
      self.defaultaf = defaultaf
      self.current = 0

    def __iter__(self):
      return self

    def next(self):
      if self.current < self.numSuites:
        suite = {}
        for key, values in self.settings.items():
          if len(values) > 1:
            suite[key] = values[self.current]
          elif len(values) == 1:
            suite[key] = values[0]
          elif key is not "activationfn":
            suite[key] = DEFAULT[key]
        if len(self.settings["activationfn"]) == 0:
          numLayer = len(suite["layer_units"])
          suite["activationfn"] = ["identity"] + [self.defaultaf] * (numLayer - 1)
        self.current += 1
        return suite
      else:
        raise StopIteration


def _assertValidSetting(fn, setting, numSuites):
  """ Helper method for checking if the setting is valid."""
  setting = dict(setting)
  if len(setting["layer_units"]) == 0:
    setting["layer_units"] = [ DEFAULT["layer_units"] ]
  for key, values in setting.items():
    if len(values) == 0 and key in ["X", "y"]:
      raise ValueError("(%s) Both the input and output trainning data must be set." % fn)
    elif len(values) > 1 and len(values) < numSuites:
      errMsg = "(%s) Two or more settings have more than one value, and the number of " % fn \
      + "values didn't match."
      raise ValueError(errMsg)
  layer_units = setting["layer_units"]
  activationfn = setting["activationfn"]
  # Check if the number of activation functions match the number of layers.
  if len(activationfn) == 1:
    for i, v in enumerate(layer_units):
      numLayer = len(v)
      if len(activationfn[0]) != numLayer:
        errMsg = "(%s) In suite #%d, the number of activation functions " % (fn, i) \
        + " (%d) didn't match the number of layers (%d)." % (len(activationfn[0]), numLayer)
        raise ValueError(errMsg)
  elif len(activationfn) > 1:
    for i in xrange(len(activationfn)):
      if len(layer_units) > 1:
        numLayer = len(layer_units[i])
      else:
        numLayer = len(layer_units)
      numActivationFn = len(activationfn[i])
      if numActivationFn != numLayer:
        errMsg = "(%s) In suite #%d, the number of activation functions " % (fn, i) \
        + " (%d) didn't match the number of layers (%d)." % (numActivationFn, numLayer)
        raise ValueError(errMsg)
  # Check if the number of input/output matches the number of input/output layer units.
  X = setting["X"]
  y = setting['y']
  if len(X) == 1:
    for i, v in enumerate(layer_units):
      if len(X[0]) != v[0]:
        errMsg = "(%s) In suite #%d, the number of input data points " % (fn, i) \
        + " (%d) didn't match the number of input layers units (%d)." % (len(X[0]), v[0])
        raise ValueError(errMsg)
  else:
    for i in xrange(len(X)):
      numInput = len(X[i])
      if len(layer_units) > 1:
        input_layer = layer_units[i][0]
      else:
        input_layer = layer_units[0][0]
      if numInput != input_layer:
        errMsg = "(%s) In suite #%d, the number of input data points " % (fn, i) \
        + " (%d) didn't match the number of input layers units (%d)." % (numInput, input_layer)
        raise ValueError(errMsg)
  if len(y) == 1:
    for i, v in enumerate(layer_units):
      if len(y[0]) != v[-1]:
        errMsg = "(%s) In suite #%d, the number of output data points " % (fn, i) \
        + " (%d) didn't match the number of output layers units (%d)." % (len(y[0]), v[-1])
        raise ValueError(errMsg)
  else:
    for i in xrange(len(y)):
      numOutput = len(y[i])
      if len(layer_units) > 1:
        output_layer = layer_units[i][-1]
      else:
        output_layer = layer_units[0][-1]
      if numOutput != output_layer:
        errMsg = "(%s) In suite #%d, the number of output data points " % (fn, i) \
        + " (%d) didn't match the number of output layers units (%d)." % (numOutput, output_layer)
        raise ValueError(errMsg)

def _assertPositiveInt(fn, name, var, layer_units=True):
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

def _assertLearnRate(fn, name, var):
  """ Helper method for checking if the user input a valid learning rate."""
  try:
    var = float(var)
  except (TypeError, ValueError):
    errMsg = "(%s) Expected %s to be a number or something that can be converted " % (fn, name) \
    + "to numerical values."
    raise TypeError(errMsg)
  if var < 0 or var > 1.0:
    raise ValueError("The learning rate should be a number between 0 and 1")

def _assertFn(fn, name, var, activationfn=True):
  """ Helper method for checking if the user input is a valid activation or error function name."""
  assertType(fn, name, var, str)
  error = False
  if activationfn and not af.contains(var):
    error = True
  elif not activationfn and not ef.contains(var):
    error = True
  if error:
    if activationfn:
      errMsg = "%s is not a valid activation function name." % var
    else:
      errMsg = "%s is not a valid error function name." % var
    raise ValueError(errMsg)

def _assertTrainningData(fn, name, var, input=True):
  """ Helper method for checking if the user input is valid input/output data."""
  if (type(var) is not np.ndarray) and (type(var) is not np.matrix):
    dataType = "input trainning data" if input else "expected output data"
    raise TypeError("(%s) Expected %s (%s) to be numpy array or matrix." % (fn, name, dataType))

def _assertValidKey(fn, key):
  """ Helper method for checking if the key corresponds to a setting."""
  assertType(fn, "key", key, str)
  if key not in validKeys:
    raise ValueError("(%s) Unknown setting: %s." % (fn, key))

def _checkLayerUnits(fn, values):
  """ Helper method for making sure the values are the correct format for the layer units."""
  if hasattr(values, "__len__") and len(values) > 0:
    newValues = []
    # Multiple values for layer units, ie. list of lists of int
    if hasattr(values[0], "__len__"):
      for value in values:
        temp = []
        for unit in value:
          _assertPositiveInt(fn, "layer_units", unit)
          temp.append(int(unit))
        newValues.append(temp)
    # One value for layer units, ie. list of int
    else:
      temp = []
      for unit in values:
        _assertPositiveInt(fn, "layer_units", unit)
        temp.append(int(unit))
      newValues.append(temp)
  else:
    raise TypeError("(%s) Expected layer_units to be a list of int or a list of list of int." % fn)
  return newValues

checkValues["layer_units"] = _checkLayerUnits

def _checkActivationFn(fn, values):
  """ Helper method for making sure the activation function names are valid."""
  if (type(values) == list or type(values) == tuple) and len(values) > 0:
    newValues = []
    # Multiple values for activation functions, ie. list of lists of str
    if (type(values[0]) == list or type(values[0]) == tuple) and len(values[0]) > 0:
      for value in values:
        temp = []
        for name in value:
          _assertFn(fn, "activationfn", name)
          temp.append(name)
        newValues.append(temp)
    # One value for activation functions, ie. list of str
    else:
      temp = []
      for name in values:
        _assertFn(fn, "activationfn", name)
        temp.append(name)
      newValues.append(temp)
  else:
    raise TypeError("(%s) Expected activationfn to be a list of str or a list of list of str." % fn)
  return newValues

checkValues["activationfn"] = _checkActivationFn

def _checkBias(fn, values):
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

checkValues["bias"] = _checkBias

def _checkErrFn(fn, values):
  """ Helper method for checking if error function is valid."""
  newValues = []
  if (type(values) == list or type(values) == tuple) and len(values) > 0:
    for value in values:
      _assertFn(fn, "errfn", value, False)
      newValues.append(value)
  else:
    _assertFn(fn, "errfn", values, False)
    newValues.append(values)
  return newValues

checkValues["errfn"] = _checkErrFn

def _checkLearnRate(fn, values):
  """ Helper method for checking if learning rate is valid."""
  newValues = []
  if hasattr(values, "__len__") and len(values) > 0:
    for value in values:
      _assertLearnRate(fn, "learn_rate", value)
      newValues.append(value)
  else:
    _assertLearnRate(fn, "learn_rate", values)
    newValues.append(values)
  return newValues

checkValues["learn_rate"] = _checkLearnRate

def _checkIterations(fn, values):
  """ Helper method for checking if number of iterations is valid."""
  newValues = []
  if hasattr(values, "__len__") and len(values) > 0:
    for value in values:
      _assertPositiveInt(fn, "iterations", value, False)
      newValues.append(value)
  else:
    _assertPositiveInt(fn, "iterations", values, False)
    newValues.append(values)
  return newValues

checkValues["iterations"] = _checkIterations

def _checkData(fn, values, input):
  """ Helper method for checking if input/output trainning data is valid."""
  newValues = []
  if (type(values) == list or type(values) == tuple) and len(values) > 0:
    for value in values:
      _assertTrainningData(fn, "X" if input else "y", value, input)
      newValues.append(value)
  else:
    _assertTrainningData(fn, "X" if input else "y", values, input)
    newValues.append(values)
  return newValues

def _checkX(fn, values):
  return _checkData(fn, values, True)

checkValues["X"] = _checkX

def _checkY(fn, values):
  return _checkData(fn, values, False)

checkValues["y"] = _checkY


def _checkMinimizer(fn, values):
  """ Helper method for minimizer."""
  newValues = []
  if hasattr(values, "__len__") and len(values) > 0:
    newValues = values
  else:
    if values == None or values == []:
      raise TypeError("(%s) Expected minimizer to be a function." % fn)
    newValues.append(values)
  return newValues

checkValues["minimizer"] = _checkMinimizer