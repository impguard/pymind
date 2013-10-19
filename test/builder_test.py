import numpy as np
from pymind.builder import Builder
from pymind.activationfn import identity, sigmoid

def testBuilderSetBias():
  """ Test the bias setter for Builder. """
  b = Builder()
  assert b.bias == False, "The initial bias should be False for Builder."
  b.setBias(True)
  assert b.bias == True, "Bias should be True after being set to True."
  b.setBias(True)
  assert b.bias == True, "Bias should be True after being set to True."
  b.setBias(False)
  assert b.bias == False, "Bias should be False after being set to False."
  b.setBias(False)
  assert b.bias == False, "Bias should be False after being set to False."

def testBuilderSetInput1():
  """ Test the setter for number of input layer units. """
  b = Builder()
  assert b.input_units == 0, "Initially, input units should be 0 but is %d" % b.input_units
  b.setInputLayer(10)
  assert b.input_units == 10, "Input units should be 10 but is %d" % b.input_units
  assert b.input_activationfnName == b.defaultfn, "Input activation function name should be %s, but is %d" % (b.defaultfn, b.input_activationfnName)
  b.setInputLayer(5)
  assert b.input_units == 5, "Input units should be 5 but is %d" % b.input_units
  assert b.input_activationfnName == b.defaultfn, "Input activation function name should be %s, but is %d" % (b.defaultfn, b.input_activationfnName)
  # Exceptions should be thrown when passed in non positive integer for input layer units.
  np.testing.assert_raises(ValueError, b.setInputLayer, -1)
  np.testing.assert_raises(ValueError, b.setInputLayer, 0)
  np.testing.assert_raises(ValueError, b.setInputLayer, 1.6)

def testBuilderSetInput2():
  """ Test the setter for input layer activation functions. """
  b = Builder()

  b.setInputLayer(activationfnName="identity")
  assert b.input_units == 0, "Input units should be 0 but is %d" % b.input_units
  assert b.input_activationfnName == "identity", "The input activation function should be %s but is %s" % ("identity", b.input_activationfnName)

  b.setInputLayer(input_units=10, activationfnName="sigmoid")
  assert b.input_units == 10, "Input units should be 10 but is %d" % b.input_units
  assert b.input_activationfnName == "sigmoid", "The input activation function should be %s but is %s" % ("sigmoid", b.input_activationfnName)

  b.setInputLayer(input_units=5, activationfnName="identity")
  assert b.input_units == 5, "Input units should be 5 but is %d" % b.input_units
  assert b.input_activationfnName == "identity", "The input activation function should be %s but is %s" % ("identity", b.input_activationfnName)

  b.setInputLayer(activationfnName="sigmoid")
  assert b.input_units == 5, "Input units should be 5 but is %d" % b.input_units
  assert b.input_activationfnName == "sigmoid", "The input activation function should be %s but is %s" % ("sigmoid", b.input_activationfnName)

  np.testing.assert_raises(ValueError, b.setInputLayer, None, "test")
  np.testing.assert_raises(ValueError, b.setInputLayer, 10, "test")
  np.testing.assert_raises(TypeError, b.setInputLayer, None, None)

def testBuilderSetOutput1():
  """ Test the setter for number of output layer units. """
  b = Builder()
  assert b.output_units == 0, "Initially, output units should be 0 but is %d" % b.output_units
  b.setOutputLayer(10)
  assert b.output_units == 10, "Output units should be 10 but is %d" % b.output_units
  assert b.output_activationfnName == b.defaultfn, "Output activation function name should be %s, but is %d" % (b.defaultfn, b.output_activationfnName)
  b.setOutputLayer(5)
  assert b.output_units == 5, "Output units should be 5 but is %d" % b.output_units
  assert b.output_activationfnName == b.defaultfn, "Output activation function name should be %s, but is %d" % (b.defaultfn, b.output_activationfnName)
  # Exceptions should be thrown when passed in non positive integer for output layer units.
  np.testing.assert_raises(ValueError, b.setOutputLayer, -1)
  np.testing.assert_raises(ValueError, b.setOutputLayer, 0)
  np.testing.assert_raises(ValueError, b.setOutputLayer, 1.6)

def testBuilderSetOutput2():
  """ Test the setter for output layer activation functions. """
  b = Builder()

  b.setOutputLayer(activationfnName="identity")
  assert b.output_units == 0, "Output units should be 0 but is %d" % b.output_units
  assert b.output_activationfnName == "identity", "The output activation function should be %s but is %s" % ("identity", b.output_activationfnName)

  b.setOutputLayer(output_units=10, activationfnName="sigmoid")
  assert b.output_units == 10, "Output units should be 10 but is %d" % b.output_units
  assert b.output_activationfnName == "sigmoid", "The output activation function should be %s but is %s" % ("sigmoid", b.output_activationfnName)

  b.setOutputLayer(output_units=5, activationfnName="identity")
  assert b.output_units == 5, "Output units should be 5 but is %d" % b.output_units
  assert b.output_activationfnName == "identity", "The output activation function should be %s but is %s" % ("identity", b.output_activationfnName)

  b.setOutputLayer(activationfnName="sigmoid")
  assert b.output_units == 5, "Output units should be 5 but is %d" % b.output_units
  assert b.output_activationfnName == "sigmoid", "The output activation function should be %s but is %s" % ("sigmoid", b.output_activationfnName)

  np.testing.assert_raises(ValueError, b.setOutputLayer, None, "test")
  np.testing.assert_raises(ValueError, b.setOutputLayer, 10, "test")
  np.testing.assert_raises(TypeError, b.setOutputLayer, None, None)

def testBuilderAddHidden():
  """ Test the adder for hidden layers."""
  b = Builder()
  exp_units = []
  exp_fnName = []

  assert b.hidden_units == exp_units, "Should not have any hidden layer."
  assert b.hidden_activationfnNames == exp_fnName, "Should not have any hidden layer."

  b.addHiddenLayers(1, "sigmoid")
  exp_units = [1]
  exp_fnName = ["sigmoid"]
  assert b.hidden_units == exp_units, "Should only have %d hidden layer(s)." % len(exp_units)
  assert b.hidden_activationfnNames == exp_fnName, "Should only have %d hidden layer." % len(exp_fnName)

  b.addHiddenLayers(2)
  exp_units.append(2)
  exp_fnName.append(b.defaultfn)
  assert b.hidden_units == exp_units, "Should only have %d hidden layer(s)." % len(exp_units)
  assert b.hidden_activationfnNames == exp_fnName, "Should only have %d hidden layer." % len(exp_fnName)

  b.addHiddenLayers([3,4,5], ["identity", "sigmoid", "sigmoid"])
  exp_units += [3, 4, 5]
  exp_fnName += ["identity", "sigmoid", "sigmoid"]
  assert b.hidden_units == exp_units, "Should only have %d hidden layer(s)." % len(exp_units)
  assert b.hidden_activationfnNames == exp_fnName, "Should only have %d hidden layer." % len(exp_fnName)

  np.testing.assert_raises(ValueError, b.addHiddenLayers, [10,10], ["identity", "identity", "identity"])

def testBuilderSetHidden():
  """ Test the setter for hidden layers."""
  b = Builder()

  np.testing.assert_raises(IndexError, b.setHiddenLayer, 1, 50)

  exp_units = [1, 2, 3]
  exp_fnName = ["identity", "sigmoid", "identity"]

  b.addHiddenLayers(exp_units, exp_fnName)

  b.setHiddenLayer(1, 10)
  exp_units[1] = 10
  assert b.hidden_units == exp_units, "Hidden layer units: \nExpected -> %r\nActual -> %r" % (exp_units, b.hidden_units)
  assert b.hidden_activationfnNames == exp_fnName, "Hidden layer activation functions: \nExpected -> %r\nActual -> %r" % (exp_fnName, b.hidden_activationfnNames)

  b.setHiddenLayer(-1, 10, "sigmoid")
  exp_units[-1] = 10
  exp_fnName[-1] = "sigmoid"
  assert b.hidden_units == exp_units, "Hidden layer units: \nExpected -> %r\nActual -> %r" % (exp_units, b.hidden_units)
  assert b.hidden_activationfnNames == exp_fnName, "Hidden layer activation functions: \nExpected -> %r\nActual -> %r" % (exp_fnName, b.hidden_activationfnNames)
  
  np.testing.assert_raises(IndexError, b.setHiddenLayer, 3, 50)

def testBuilderInsertHidden():
  """ Test the insert method for hidden layer."""
  b = Builder()
  exp_units = [1]
  exp_fnName = [b.defaultfn]

  b.insertHiddenLayer(10, 1)
  assert b.hidden_units == exp_units, "Hidden layer units: \nExpected -> %r\nActual -> %r" % (exp_units, b.hidden_units)
  assert b.hidden_activationfnNames == exp_fnName, "Hidden layer activation functions: \nExpected -> %r\nActual -> %r" % (exp_fnName, b.hidden_activationfnNames)

  exp_units.append(2)
  exp_fnName.append("identity")

  b.insertHiddenLayer(10, 2, "identity")
  assert b.hidden_units == exp_units, "Hidden layer units: \nExpected -> %r\nActual -> %r" % (exp_units, b.hidden_units)
  assert b.hidden_activationfnNames == exp_fnName, "Hidden layer activation functions: \nExpected -> %r\nActual -> %r" % (exp_fnName, b.hidden_activationfnNames)

  exp_units.insert(0, 5)
  exp_fnName.insert(0, "sigmoid")

  b.insertHiddenLayer(0, 5, "sigmoid")
  assert b.hidden_units == exp_units, "Hidden layer units: \nExpected -> %r\nActual -> %r" % (exp_units, b.hidden_units)
  assert b.hidden_activationfnNames == exp_fnName, "Hidden layer activation functions: \nExpected -> %r\nActual -> %r" % (exp_fnName, b.hidden_activationfnNames)

  np.testing.assert_raises(TypeError, b.insertHiddenLayer, 1.0, 1)

def testBuilderRemoveHidden():
  """ Test removing hidden layer."""
  b = Builder()

  np.testing.assert_raises(IndexError, b.removeHiddenLayer, 1)
  np.testing.assert_raises(IndexError, b.removeHiddenLayer, 0)
  np.testing.assert_raises(IndexError, b.removeHiddenLayer, -1)

  b.addHiddenLayers([1,2,3,4,5])
  b.removeHiddenLayer(0)
  exp_units = [2,3,4,5]
  exp_fnName = [b.defaultfn] * 4
  assert b.hidden_units == exp_units, "Hidden layer units: \nExpected -> %r\nActual -> %r" % (exp_units, b.hidden_units)
  assert b.hidden_activationfnNames == exp_fnName, "Hidden layer activation functions: \nExpected -> %r\nActual -> %r" % (exp_fnName, b.hidden_activationfnNames)

  np.testing.assert_raises(TypeError, b.removeHiddenLayer, 1.5)

  b.removeHiddenLayer(2)
  exp_units.pop(2)
  exp_fnName.pop(2)
  assert b.hidden_units == exp_units, "Hidden layer units: \nExpected -> %r\nActual -> %r" % (exp_units, b.hidden_units)
  assert b.hidden_activationfnNames == exp_fnName, "Hidden layer activation functions: \nExpected -> %r\nActual -> %r" % (exp_fnName, b.hidden_activationfnNames)

  np.testing.assert_raises(IndexError, b.removeHiddenLayer, 3)

def testBuilderBuild():
  """ Test the construction of a NeuralNetwork by Builder."""
  b = Builder()

  np.testing.assert_raises(ValueError, b.build)

  params={
    "input_units": 5,
    "output_units": 3,
    "hidden_units": 4,
    "activationfn": [identity, sigmoid, sigmoid],
    "bias": True
  }

  b.setInputLayer(params["input_units"], "identity")
  b.addHiddenLayers(params["hidden_units"], "sigmoid")
  b.setOutputLayer(params["output_units"], "sigmoid")
  b.setBias(params["bias"])

  nnetwork = b.build()
  assert nnetwork.input_units == params["input_units"], "Input units should be %d" % params["input_units"]
  assert nnetwork.hidden_units == [params["hidden_units"]], "Hidden units should be %d" % params["hidden_units"]
  assert nnetwork.output_units == params["output_units"], "Output units should be %d" % params["output_units"]
  assert nnetwork.activationfn == params["activationfn"], "Activation functions didn't match."
  assert nnetwork.bias == params['bias'], "Bias didn't match."

  b.reset()
  np.testing.assert_raises(ValueError, b.build)

  params = {
    "input_units": 5,
    "output_units": 3,
    "hidden_units": [4, 4, 3],
    "activationfn": [identity, sigmoid, sigmoid, sigmoid, sigmoid],
    "bias": False
  }

  b.setInputLayer(params["input_units"], "identity")
  b.addHiddenLayers(params["hidden_units"], ["sigmoid"]*3)
  b.setOutputLayer(params["output_units"], "sigmoid")
  b.setBias(params["bias"])

  nnetwork = b.build()
  assert nnetwork.input_units == params["input_units"], "Input units didn't match."
  assert nnetwork.hidden_units == params["hidden_units"], "Hidden units didn't match."
  assert nnetwork.output_units == params["output_units"], "Output units didn't match."
  assert nnetwork.activationfn == params["activationfn"], "Activation functions didn't match."
  assert nnetwork.bias == params['bias'], "Bias didn't match."

def testBuilderAddActivationFn1():
  """ Test adding user's own activation functions."""
  b = Builder()
  class linear(identity):
    @classmethod
    def _calc(cls, v):
      return v
    @classmethod
    def _grad(cls, v):
     raise 1
  class quadratic(identity):
    @classmethod
    def _calc(cls, v):
      return v*v
    @classmethod
    def _grad(cls, v):
     raise 2*v
  name2fn = {"linear": linear, "quadratic": quadratic}
  b.addActivationFunctions(name2fn)

  exp = {"identity": identity, "sigmoid": sigmoid, "linear": linear, "quadratic": quadratic}
  assert b.name2fn == exp, "Dictionary of activation functions did not match."

def testBuilderAddActivationFn2():
  """ Test adding user's own activation functions."""
  b = Builder()
  class linear(identity):
    @classmethod
    def _calc(cls, v):
      return v
    @classmethod
    def _grad(cls, v):
     raise 1
  np.testing.assert_raises(TypeError, b.addActivationFunctions, {1:linear})

def testBuilderSetDefaultfn():
  """ Test the setter for default activation function."""
  b = Builder()
  class linear(identity):
    @classmethod
    def _calc(cls, v):
      return v
    @classmethod
    def _grad(cls, v):
     raise 1
  class quadratic(identity):
    @classmethod
    def _calc(cls, v):
      return v*v
    @classmethod
    def _grad(cls, v):
     raise 2*v
  name2fn = {"linear": linear, "quadratic": quadratic}
  b.addActivationFunctions(name2fn)

  b.setDefaultfn("linear")
  assert b.defaultfn == "linear", "The default activation function should be %s, not %s" % ("linear", b.defaultfn)

  b.setDefaultfn("identity")
  assert b.defaultfn == "identity", "The default activation function should be %s, not %s" % ("identity", b.defaultfn)

  np.testing.assert_raises(TypeError, b.setDefaultfn, 10)
  np.testing.assert_raises(ValueError, b.setDefaultfn, "Cubic") 