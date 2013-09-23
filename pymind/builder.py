from components import NeuralNetwork
from activationfn import identity, sigmoid

class Builder(object):

  def __init__(self):
    # --- Initialize default setting  ---- #
    self.defaultfn = "sigmoid"
    self.input_units = 0
    self.output_units = 0
    self.hidden_units = []
    self.input_activationfnName = self.defaultfn
    self.output_activationfnName = self.defaultfn
    self.hidden_activationfnNames = []
    self.name2fn = {"identity": identity, "sigmoid": sigmoid}
    self.bias = False

  def setBias(self, bias):
    """ Set the bias to be true or false.

    Arguments:
    bias (boolean): The bias of the neural network that can be constructed by this Builder.

    Returns:
    The Builder object itself
    """
    if type(bias) is not bool:
      # bias must be a boolean
      raise TypeError("Bias must be True or False.")
    self.bias = bias
    return self

  def setInputLayer(self, input_units=None, activationfnName=None):
    """ Set the number of input layer units (and/or activation fucntion) of the input layer.

    Arguments:
    input_units (int): A positive integer representing the number of input layer units. No change to number of input units if nothing is passed in.
    activationfnName (string): A string representing the activation function for the input layer. No change to input layer activation function if nothing is passed in.

    Returns:
    The Builder object itself.
    """
    self._setLayer(layer="INPUT", units=input_units, activationfnName=activationfnName)
    return self

  def setOutputLayer(self, output_units=None, activationfnName=None):
    """ Set the number of output layer units (and/or activation fucntion) of the output layer.

    Arguments:
    output_units (int): A positive integer representing the number of output layer units. No change to number of output units if nothing is passed in.
   activationfnName (string): A string representing the activation function for the output layer. No change to output layer activation function if nothing is passed in.

    Returns:
    The Builder object itself.
    """
    self._setLayer(layer="OUTPUT", units=output_units, activationfnName=activationfnName)
    return self

  def setHiddenLayer(self, index, hidden_units=None, activationfnName=None):
    """ For a hidden layer of the indicated index, set its number of layer units (and/or activation function).

    Arguments:
    index (int): An integer indicating which hidden layer to be changed. Indexing starts from 0.
    hidden_units (int): A positive integer representing the number of hidden layer units. No change to number of hidden units if nothing is passed in.
    activationfnName (string): A string representing the activation function for the hidden layer. No change to hidden layer activation function if nothing is passed in.

    Returns:
    The Builder object itself.
    """
    self._setLayer(layer="HIDDEN", units=hidden_units, activationfnName=activationfnName, index=index)
    return self

  def _setLayer(self, layer, units, activationfnName, index=0):
    if units is not None and (type(units) is not int or units <= 0):
      # Raise exception if number of layer units is not a positive integer
      raise ValueError("The number of layer units must be a positive integer.")
    if activationfnName is not None and activationfnName not in self.activationfnNames():
      # Raise exception if the activation function name is not in the set of acceptable names of this Builder
      raise ValueError("The name '" + str(activationfnName) + "' does not match any valid activation function. This Builder object has the following valid activation function names: " + str(self.activationfnNames()))
    if units is None and activationfnName is None:
      # Not enough arguments
      if layer == "INPUT" or layer == "OUTPUT":
        raise TypeError("Expected at least 1 argument, got 0")
      elif layer == "HIDDEN":
        raise TypeError("Expected at least 2 argument, got 1")
    if layer == "INPUT":

      if units:
        self.input_units = units

      if activationfnName:
        self.input_activationfnName = activationfnName

    elif layer == "OUTPUT":

      if units:
        self.output_units = units

      if activationfnName:
        self.output_activationfnName = activationfnName
    elif layer == "HIDDEN":

      if len(self.hidden_units) == 0:
        #  No hidden layer to be changed
        raise IndexError("No hidden layer to be changed.")

      if type(index) is not int:
        # Cannot have non-integer index
        raise TypeError("Index must be an integer.")

      if index >= len(self.hidden_units):
        # Raise exception if the index is out of bound
        raise IndexError("Index out of bound. Only " + str(len(self.hidden_units)) + " hidden layers, but try to change the " + str(index+1) + "th layer.")

      if units:
        self.hidden_units[index] = units

      if activationfnName:
        self.hidden_activationfnNames[index] = activationfnName
    else:
      raise TypeError("Unknown layer type: " + str(layer))

  def addHiddenLayers(self, hidden_units, activationfnNames=None):
    """ Add new hidden layer(s).

    Arguments:
    hidden_units (int|list:int): A positive integer representing the number of hidden layer units; A list of positive integers representing the number of hidden layer units per hidden layer.
    activationfnNames (string|list:string): A string representing the activation function for the hidden layer. See activationfnNames() for accpetable function names; A list of strings representing the activation functions for each hidden layer, number of functions must match that of hidden layers. The default activation function name is used if this parameter receive no input. 

    Returns: 
    The Builder object itself.
    """
    if type(hidden_units) is int:
      # Convert integer to one element list
      hidden_units = [hidden_units]

    if not activationfnNames:
      # Use the default activation function if nothing is passed in for activationfnNames
      activationfnNames = [self.defaultfn] * len(hidden_units)

    if type(activationfnNames) is str:
      # Convert string to one element list
      activationfnNames = [activationfnNames]

    if len(hidden_units) != len(activationfnNames):
      # Raise exception if number of hidden layer units doesn't match the number of activation functions
      raise ValueError("The number of hidden layers must match the number of activation functions.")

    # Add the hidden layers
    for i in xrange(len(hidden_units)):
      if type(hidden_units[i]) is not int or hidden_units[i] <= 0:
        # Raise exception if the hidden layer unit is not a positive integer
        raise ValueError("The number of hidden layer units must be a positive integer.")

      if type(activationfnNames[i]) is not str:
        # Raise exception if the activation function name is not a string representing the name of an activation function
        raise TypeError("The name of the activation function should be a string.")

      if activationfnNames[i] not in self.activationfnNames():
        # Raise exception if the activation function name is not in the set of acceptable names of this Builder
       raise ValueError("The name '" + str(activationfnNames[i]) + "' does not match any valid activation function. This Builder object has the following valid activation function names: " + str(self.activationfnNames()))

      # Store the information about the new hidden layer
      self.hidden_units.append(hidden_units[i])
      self.hidden_activationfnNames.append(activationfnNames[i])
    return self

  def insertHiddenLayer(self, index, hidden_unit, activationfnName=None):
    """ Insert a hidden layer at the given index.

    Arguments:
    index (int): an integer indicating at which index the hidden layer will be inserted into.
    hidden_unit (int): a positive integer representing the number of hidden layer units.
    activationfnName (string): The name of the activation function of the hidden layer. Default function name will be used if no argument passed in.

    Returns:
    The Builder object itself.
    """
    if activationfnName == None:
      activationfnName = self.defaultfn
    if type(index) is not int:
      # Cannot have non-integer index
      raise TypeError("Index must be an integer.")
    if type(hidden_unit) is not int or hidden_unit <= 0:
      # Raise exception if number of hidden layer units is not a positive integer
      raise ValueError("The number of hidden layer units must be a positive integer.")
    if activationfnName not in self.activationfnNames():
      # Raise exception if the activation function name is not in the set of acceptable names of this Builder
      raise ValueError("The name '" + str(activationfnName) + "' does not match any valid activation function. This Builder object has the following valid activation function names: " + str(self.activationfnNames()))
    # Insert the hidden layer
    self.hidden_units.insert(index, hidden_unit)
    self.hidden_activationfnNames.insert(index, activationfnName)
    return self

  def removeHiddenLayer(self, index):
    """ Remove the hidden layers specified by the given index.

    Arguments:
    index (int): an integer indicating which hidden layer is to be removed. Index starts from 0.

    Returns:
    The Builder object itself.
    """
    if len(self.hidden_units) == 0:
      # Cannot remove from an empty list
      raise IndexError("No hidden layer to be removed.")
    if type(index) is not int:
      # Cannot have non-integer index
      raise TypeError("Index must be an integer.")
    if index >= len(self.hidden_units):
      # Raise exception if the index is out of bound
      raise IndexError("Index out of bound. Only " + str(len(self.hidden_units)) + " hidden layers, but try to remove the " + str(index+1) + "th layer.")
    # Remove the hidden layer
    self.hidden_units.pop(index)
    self.hidden_activationfnNames.pop(index)
    return self

  def activationfnNames(self):
    """ Return the set of strings representing the activation functions accpeted by this Builder object.
    """
    return self.name2fn.keys()

  def getInputLayer(self):
    """ Return the number of input layer units and the activation function.

    Arguments:
    None

    Returns:
    A tuple containing the number of input layer units and the activation function.
    """
    return self.input_units, self.input_activationfnName

  def getOutputLayer(self):
    """ Return the number of output layer units and the activation function.

    Arguments:
    None

    Returns:
    A tuple containing the number of output layer units and the activation function.
    """
    return self.output_units, self.output_activationfnName

  def getHiddenLayer(self, index=None):
    """ For the indicated hidden layer, return the number of hidden layer units and the activation function. All the hidden layers are returned if no index is passed in.

    Arguments:
    None

    Returns:
    A tuple containing the number of output layer units and the activation function; or a tuple of two lists, each entry in the list correspond to the number of layer units and activation function for a hidden layer. 
    """
    if len(self.hidden_units) == 0:
      # Cannot remove from an empty list
      raise IndexError("No hidden layer to look at.")
    if type(index) is not int:
      # Cannot have non-integer index
      raise TypeError("Index must be an integer.")
    if index >= len(self.hidden_units):
      # Raise exception if the index is out of bound
      raise IndexError("Index out of bound. Only " + str(len(self.hidden_units)) + " hidden layers, but try to look at the " + str(index+1) + "th layer.")
    if index:
      return self.hidden_units[i], self.hidden_activationfnNames[i]
    else:
      return self.hidden_units, self.hidden_activationfnNames

  def display(self):
    """ Display the current setting of the Builder. Warnings will appear if the number of units for the input and/or output layer is 0.

    Arguments: None
    Returns: None
    """
    if self.input_units == 0:
      # Cannot construct neural network with zero input units
      print "Warning: Must have at least one input units."
    if self.output_units == 0:
      # Cannot construct neural network with zero outut units
      print "Warning: Must have at least one output units."

    # Brief summary of the current settings of the Builder
    summary = "Have " + str(len(self.hidden_units)) + " hidden layers and has "
    summary += "bias." if self.bias else "no bias."
    print summary
    
    print "Input Layer: %5d, %s" % (self.input_units, self.input_activationfnName)

    if len(self.hidden_units) > 0:
      print "\nHidden Layers:"
    for i in xrange(len(self.hidden_units)):
      print "%18d, %s" % (self.hidden_units[i], self.hidden_activationfnNames[i])

    print "\nOutput Layer: %4d, %s" % (self.output_units, self.output_activationfnName)

  def reset(self):
    """ Revert the Builder to the initial setting, except the default activation function and name-to-function dictionary if they are altered.

    Arguments:
    None

    Returns:
    The Builder object itself
    """
    self.input_units = 0
    self.output_units = 0
    self.hidden_units = []
    self.input_activationfnName = self.defaultfn
    self.output_activationfnName = self.defaultfn
    self.hidden_activationfnNames = []
    self.bias = False
    return self


  def build(self):
    """ Construct a PyMind neural network based on the setting of the Builder.

    Arguments:
    None

    Returns:
    A PyMind NeuralNetwork object.
    """
    if self.input_units == 0:
      # Cannot construct neural network with zero input units
      raise ValueError("Must have at least one input units.")
    if self.output_units == 0:
      # Cannot construct neural network with zero outut units
      raise ValueError("Must have at least one output units.")

    fnNames = [self.input_activationfnName] + self.hidden_activationfnNames + [self.output_activationfnName]
    # Get the activation functions based on the names inputted by user
    activationfn = [self.name2fn[name] for name in fnNames]
    # Setting up the parameters that is required by NeuralNetwork constructor
    params = {
    "input_units": self.input_units, 
    "output_units": self.output_units, 
    "hidden_units": self.hidden_units, 
    "activationfn": activationfn, 
    "bias": self.bias
    }
    return NeuralNetwork(params)