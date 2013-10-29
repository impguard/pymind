import numpy as np
from pymind.components.nnbuilder import Builder, DEFAULT
import pymind.activationfn as af
import pymind.errfn as ef

def testBuilderConstructor1():
  """ Testing the default (no arguments) constructor of the Builder."""
  b = Builder()
  info = b.getSetting()
  # Settings for the neural network
  assert info['layer_units'] == DEFAULT['layer_units'], "Layer units didn't match default setting."
  assert info['activationfn'] == [], "Activation functions didn't match default \
  setting."
  assert info['bias'] == DEFAULT['bias'], "Bias didn't match default setting."
  # Settings for the trainning
  assert info['errfn'] == DEFAULT['errfn'], "Error function didn't match default setting."
  assert info['learn_rate'] == DEFAULT['learn_rate'], "Learning rate didn't match default setting."
  assert info['minimizer'] == DEFAULT['minimizer'], "Minimizer didn't match default setting."
  assert info['iterations'] == DEFAULT['iterations'], "Number of trainning iterations didn't match \
  default setting."
  assert info['X'] == [], "Input trainning data (X) should be []."
  assert info['y'] == [], "Expected output data (y) shoulde be []."

def testBuilderConstructor2():
  """ Testing the constructor of the Builder with simple argument."""
  dummyMinimizer = lambda x: x
  setting = {
    "minimizer": dummyMinimizer,
    "bias": False,
    "activationfn": ["sigmoid", "identity", "sigmoid"],
    "iterations": 1337,
    "learn_rate": 1.0,
    "layer_units": [10, 20, 1],
    "errfn": "logitError"
  }
  b = Builder(setting)
  info = b.getSetting()
  # Settings for the neural network
  assert info['layer_units'] == [10,20,1], "Layer units didn't match user setting."
  assert info['activationfn'] == ["sigmoid", "identity", "sigmoid"], "Activation functions didn't \
  match user setting."
  assert info['bias'] == False, "Bias didn't match user setting."
  # Settings for the trainning
  assert info['errfn'] == "logitError", "Error function didn't match user setting."
  assert info['learn_rate'] == 1.0, "Learning rate didn't match user setting."
  assert info['minimizer'] == dummyMinimizer, "Minimizer didn't match user setting."
  assert info['iterations'] == 1337, "Number of trainning iterations didn't match user setting."
  assert info['X'] == [], "Input trainning data (X) should be []."
  assert info['y'] == [], "Expected output data (y) shoulde be []."

def testBuilderConstructor3():
  """ Testing the constructor of the Builder with another simple argument."""
  dummyMinimizer = lambda x: x
  setting = {
    "minimizer": dummyMinimizer,
    "bias": False,
    "layer_units": [10, 20, 1],
    "errfn": "logitError",
    "X": np.arange(15).reshape(5,3),
    "y": np.array([[1,2,3],[4,5,6]])
  }
  X = np.arange(15).reshape(5,3)
  y = np.arange(6).reshape(2,3)
  y[0] = [1, 2, 3]
  y[1] = [4, 5, 6]
  b = Builder(setting)
  info = b.getSetting()
  # Settings for the neural network
  assert info['layer_units'] == [10,20,1], "Layer units didn't match user setting."
  assert info['activationfn'] == [], "Activation functions didn't match user \
  setting."
  assert info['bias'] == False, "Bias didn't match user setting."
  # Settings for the trainning
  assert info['errfn'] == "logitError", "Error function didn't match user setting."
  assert info['learn_rate'] == DEFAULT['learn_rate'], "Learning rate didn't match user setting."
  assert info['minimizer'] == dummyMinimizer, "Minimizer didn't match user setting."
  assert info['iterations'] == DEFAULT["iterations"], "Number of trainning iterations didn't match \
  user setting."
  assert np.array_equal(info['X'], X), "Input trainning data (X) should \
  match user setting."
  assert np.array_equal(info['y'], y), "Expected output data (y) shoulde be None."

def testBuilderConstructor4():
  """ Testing the constructor of the Builder with a more complex argument."""
  dummyMinimizer = lambda x: x
  setting = {
    "minimizer": dummyMinimizer,
    "bias": [False, True],
    "layer_units": [[10, 20, 1], [5, 2]],
    "iterations": [1337, 10, 232, 1],
    "errfn": "logitError"
  }
  b = Builder(setting)
  info = b.getSetting()
  # Settings for the neural network
  assert info['layer_units'] == [[10,20,1], [5, 2]], "Layer units didn't match user setting."
  assert info['activationfn'] == [], "Activation functions didn't match user setting."
  assert info['bias'] == [False, True], "Bias didn't match user setting."
  # Settings for the trainning
  assert info['errfn'] == "logitError", "Error function didn't match user setting."
  assert info['learn_rate'] == DEFAULT['learn_rate'], "Learning rate didn't match user setting."
  assert info['minimizer'] == dummyMinimizer, "Minimizer didn't match user setting."
  assert info['iterations'] == [1337, 10, 232, 1], "Number of trainning iterations didn't match \
  user setting."
  assert info['X'] == [], "Input trainning data (X) should be []."
  assert info['y'] == [], "Expected output data (y) shoulde be []."

def testConstructorError():
  """ Testing the errors of the constructor."""
  try:
    b = Builder("a")
    assert False, "Should raise TypeError for non-dictionary argument for constructor."
  except TypeError:
    pass
  try:
    b = Builder({"test":1})
    assert False, "Should raise ValueError for dicitonary to contain none of the valid keys."
  except ValueError:
    pass
  try:
    b = Builder({"layer_units": 1})
    assert False, "Should raise TypeError for incorrect key-value pair."
  except TypeError:
    pass

def testGetter1():
  """ Testing the getter for one setting."""
  dummyMinimizer = lambda x: x
  setting = {
    "minimizer": dummyMinimizer,
    "bias": False,
    "layer_units": [10, 20, 1],
    "errfn": "logitError",
    "X": np.arange(15).reshape(5,3),
    "y": np.array([[1,2,3],[4,5,6]])
  }
  X = np.arange(15).reshape(5,3)
  y = np.arange(6).reshape(2,3)
  y[0] = [1, 2, 3]
  y[1] = [4, 5, 6]
  b = Builder(setting)
  info = b.getSetting()
  # Settings for the neural network
  assert b.get('layer_units') == [10,20,1], "Layer units didn't match user setting."
  assert b.get('activationfn') == [], "Activation functions didn't match user setting."
  assert b.get('bias') == False, "Bias didn't match user setting."
  # Settings for the trainning
  assert b.get('errfn') == "logitError", "Error function didn't match user setting."
  assert b.get('learn_rate') == DEFAULT['learn_rate'], "Learning rate didn't match user setting."
  assert b.get('minimizer') == dummyMinimizer, "Minimizer didn't match user setting."
  assert info['iterations'] == DEFAULT["iterations"], "Number of trainning iterations didn't match \
  user setting."
  assert np.array_equal(b.get('X'), X), "Input trainning data (X) should match user setting."
  assert np.array_equal(b.get('y'), y), "Expected output data (y) shoulde be None."

def testGetter2():
  """ Testing the getter for multiple settings."""
  dummyMinimizer = lambda x: x
  setting = {
    "minimizer": dummyMinimizer,
    "bias": [False, True],
    "layer_units": [[10, 20, 1], [5, 2]],
    "iterations": [1337, 10, 232, 1],
    "errfn": "logitError"
  }
  b = Builder(setting)
  info = b.get("bias", "errfn", "learn_rate")
  assert info[0] == [False, True], "Bias didn't matched expected setting."
  assert info[1] == "logitError", "Error function didn't matched expected setting."
  assert info[2] == DEFAULT['learn_rate'], "Learning rate didn't matched expected setting."
  info = b.get("layer_units", "learn_rate", "iterations")
  assert info[0][0] == [10, 20, 1], "Layer units of first suite didn't match expected setting."
  assert info[0][1] == [5, 2], "Layer units of second suite didn't match expected setting."
  assert info[1] == DEFAULT["learn_rate"], "Learning rate didn't match expected setting."
  assert info[2] == [1337, 10, 232, 1], "Numbers of iterations didn't match expected setting."

def testSetter1():
  """ Testing the setter for single setting."""
  b = Builder()
  b.set(layer_units=[1,10,2])
  assert b.get("layer_units") == [1, 10, 2], "Layer units didn't match expected setting."
  b.set(layer_units=[[1,2], [5,5,5]])
  assert b.get("layer_units") == [[1, 2], [5, 5, 5]], "Layer units didn't match expected setting."
  b.set(errfn="logitError")
  assert b.get("errfn") == "logitError", "Error function didn't match expected setting."
  b.set(learn_rate=[0.5,0.9,0.25])
  assert b.get("learn_rate") == [0.5,0.9,0.25], "Learning rates didn't match expected setting."
  b.set(X=np.arange(6).reshape(3,2))
  assert np.array_equal(b.get("X"), np.array([[0,1],[2,3],[4,5]])), "Input testing data didn't \
  match expected setting."
  b.set(X=np.array([[0,1,2],[3,4,5]]))
  assert np.array_equal(b.get("X"), np.arange(6).reshape(2,3)), "Input testing data didn't match \
  expected setting."
  b.set(y=[np.array([[5,7],[1,2]]), np.array([[1,2], [10,1]])])
  assert np.array_equal(b.get("y")[0], np.array([[5,7],[1,2]])), "Output data didn't match expected\
  setting."
  assert np.array_equal(b.get("y")[1], np.array([[1,2],[10,1]])), "Output data didn't match \
  expected setting."

def testSetter2():
  """ Testing the setter for multple settings."""
  b = Builder()
  b.set(layer_units=[10,10], bias=[True, True], activationfn=["sigmoid", "sigmoid"], iterations=15) 
  assert b.get("layer_units") == [10,10], "Layer units didn't match expected setting."
  assert b.get("bias") == [True, True], "Bias didn't match expected setting."
  assert b.get("activationfn") == ["sigmoid", "sigmoid"], "Activation functions didn't match \
  expected setting."
  assert b.get("iterations") == 15, "Number of iterations didn't match expected setting."
  assert b.get("learn_rate") == DEFAULT['learn_rate'], "Learning rate didn't match expected setting."
  dummyMinimizer = lambda x: x
  setting = {
    "minimizer": dummyMinimizer,
    "bias": False,
    "activationfn": ["sigmoid", "identity", "sigmoid"],
    "iterations": 1337,
    "learn_rate": 1.0,
    "layer_units": [10, 20, 1],
    "errfn": "logitError"
  }
  b.set(**setting)
  assert b.get("layer_units") == [10,20,1], "Layer units didn't match expected setting."
  assert b.get("minimizer") == dummyMinimizer, "Minimizer didn't match expected setting."
  assert b.get("iterations") == 1337, "Number of iterations didn't match expected setting."
  assert b.get("learn_rate") == 1.0, "Learning rate didn't match expected setting."

def testSetterErrors():
  """ Testing the errors of the setter."""
  b = Builder()
  try:
    b.set()
    assert False, "Should raise TypeError for no arguments."
  except TypeError:
    pass
  try:
    b.set(layer_units=[1,1,0])
    assert False, "Passing in an 0 for layer unit should raise a ValueError."
  except ValueError:
    pass
  try:
    b.set(layer_units=[1,1,-3])
    assert False, "Passing in a -3 for layer unit should raise a ValueError."
  except ValueError:
    pass
  try:
    b.set(layer_units=[1,'a',3])
    assert False, "Passing in 'a' for layer unit should raise a TypeError."
  except TypeError:
    pass
  b.set(layer_units=[2.5, np.array([1])[0]])
  try:
    b.set(activationfn=["identity", 3, "sigmoid"])
    assert False, "Should raise a TypeError for activation function name not being string."
  except TypeError:
    pass  
  try:
    b.set(activationfn=["identity", "dummy", "sigmoid"])
    assert False, "Should raise an ValueError for invalid activation function name."
  except ValueError:
    pass
  try:
    b.set(errfn=[])
    assert False, "Should raise a TypeError for error function name not being string."
  except TypeError:
    pass  
  try:
    b.set(errfn="dummy")
    assert False, "Should raise an ValueError for invalid error function name."
  except ValueError:
    pass
  try:
    b.set(bias=10)
    assert False, "Should raise TypeError for bias not being boolean."
  except TypeError:
    pass
  try:
    b.set(learn_rate=-0.5)
    assert False, "Should raise ValueError for learning rate being < 0."
  except ValueError:
    pass
  try:
    b.set(learn_rate=1.1)
    assert False, "Should raise ValueError for learning rate being > 1."
  except ValueError:
    pass
  try:
    b.set(learn_rate='a')
    assert False, "Should raise TypeError for learning rate not being a number."
  except TypeError:
    pass
  try:
    b.set(iterations=0)
    assert False, "Should raise ValueError for number of iterations being non-positive."
  except ValueError:
    pass
  try:
    b.set(iterations=-10)
    assert False, "Should raise ValueError for number of iterations being non-positive."
  except ValueError:
    pass
  try:
    b.set(iterations='a')
    assert False, "Should raise TypeError for number of iterations not being a number."
  except TypeError:
    pass
  try:
    b.set(X='a')
    assert False, "Should raise TypeError for input data not being a numpy array or matrix."
  except TypeError:
    pass
  try:
    b.set(X=[[1,2],[3,4]])
    assert False, "Should raise TypeError for input data not being a numpy array or matrix."
  except TypeError:
    pass
  try:
    b.set(y='a')
    assert False, "Should raise TypeError for output data not being a numpy array or matrix."
  except TypeError:
    pass
  try:
    b.set(y=[[1,2],[3,4]])
    assert False, "Should raise TypeError for output data not being a numpy array or matrix."
  except TypeError:
    pass

def testAppend():
  """ Testing the appender for single setting."""
  b = Builder()
  b.set(layer_units=[2,1])
  b.append(layer_units=[[10,2,4], [7,5,6,1]])
  assert b.get("layer_units") == [[2,1], [10,2,4], [7,5,6,1]], "Layer units didn't match expected \
  setting."
  b.set(iterations=[10, 15])
  b.append(iterations=70)
  assert b.get("iterations") == [10,15,70], "Numbers of iterations didn't match expected setting."
  b.set(X=np.arange(6).reshape(3,2))
  b.append(X=np.array([1,2,3]))
  assert np.array_equal(b.get("X")[0], np.array([[0,1],[2,3],[4,5]])), "Input trainning data didn't\
   match expected setting."
  assert np.array_equal(b.get("X")[1], np.array([1,2,3])), "Input trainning data didn't match \
  expected setting."

def testAppend2():
  """ Testing the appender for multiple settings."""
  b = Builder()
  b.set(layer_units=[2,1], iterations=[10, 15])
  b.append(iterations=70, layer_units=[[10,2,4], [7,5,6,1]])
  assert b.get("layer_units") == [[2,1], [10,2,4], [7,5,6,1]], "Layer units didn't match expected \
  setting."
  assert b.get("iterations") == [10,15,70], "Numbers of iterations didn't match expected setting."

def testAppendErrors():
  """ Testing the errors of the appender."""
  b = Builder()
  try:
    b.append()
    assert False, "Should raise TypeError for no arguments."
  except TypeError:
    pass
  try:
    b.append(layer_units=[1,1,0])
    assert False, "Passing in an 0 for layer unit should raise a ValueError."
  except ValueError:
    pass
  try:
    b.append(layer_units=[1,1,-3])
    assert False, "Passing in a -3 for layer unit should raise a ValueError."
  except ValueError:
    pass
  try:
    b.append(layer_units=[1,'a',3])
    assert False, "Passing in 'a' for layer unit should raise a TypeError."
  except TypeError:
    pass
  b.append(layer_units=[2.5, np.array([1])[0]])
  try:
    b.append(activationfn=["identity", 3, "sigmoid"])
    assert False, "Should raise a TypeError for activation function name not being string."
  except TypeError:
    pass  
  try:
    b.append(activationfn=["identity", "dummy", "sigmoid"])
    assert False, "Should raise an ValueError for invalid activation function name."
  except ValueError:
    pass
  try:
    b.append(errfn=[])
    assert False, "Should raise a TypeError for error function name not being string."
  except TypeError:
    pass  
  try:
    b.append(errfn="dummy")
    assert False, "Should raise an ValueError for invalid error function name."
  except ValueError:
    pass
  try:
    b.append(bias=10)
    assert False, "Should raise TypeError for bias not being boolean."
  except TypeError:
    pass
  try:
    b.append(learn_rate=-0.5)
    assert False, "Should raise ValueError for learning rate being < 0."
  except ValueError:
    pass
  try:
    b.append(learn_rate=1.1)
    assert False, "Should raise ValueError for learning rate being > 1."
  except ValueError:
    pass
  try:
    b.append(learn_rate='a')
    assert False, "Should raise TypeError for learning rate not being a number."
  except TypeError:
    pass
  try:
    b.append(iterations=0)
    assert False, "Should raise ValueError for number of iterations being non-positive."
  except ValueError:
    pass
  try:
    b.append(iterations=-10)
    assert False, "Should raise ValueError for number of iterations being non-positive."
  except ValueError:
    pass
  try:
    b.append(iterations='a')
    assert False, "Should raise TypeError for number of iterations not being a number."
  except TypeError:
    pass
  try:
    b.append(X='a')
    assert False, "Should raise TypeError for input data not being a numpy array or matrix."
  except TypeError:
    pass
  try:
    b.append(X=[[1,2],[3,4]])
    assert False, "Should raise TypeError for input data not being a numpy array or matrix."
  except TypeError:
    pass
  try:
    b.append(y='a')
    assert False, "Should raise TypeError for output data not being a numpy array or matrix."
  except TypeError:
    pass
  try:
    b.append(y=[[1,2],[3,4]])
    assert False, "Should raise TypeError for output data not being a numpy array or matrix."
  except TypeError:
    pass

def testInsert():
  """ Testing inserting setting."""
  b = Builder()
  b.set(iterations=5, learn_rate=0.5, bias=[False, False, False])
  b.insert(0, iterations=10)
  assert b.get("iterations") == [10,5], "Numbers of iterations didn't match expected setting."
  b.insert(1, learn_rate=1.0)
  assert b.get("learn_rate") == [0.5,1.0], "Numbers of iterations didn't match expected setting."
  b.insert(2, bias=True)
  assert b.get("bias") == [False,False,True,False], "Bias didn't match expected setting."
  b.insert(4, bias=True)
  assert b.get("bias") == [False,False,True,False,True], "Bias didn't match expected setting."
  b.insert(1, iterations=[1,2,3])
  assert b.get("iterations") == [10,1,2,3,5], "Numbers of iterations didn't match expected setting."
  b.insert(2, learn_rate=[0.1,0.2])
  assert b.get("learn_rate") == [0.5, 1.0, 0.1, 0.2], "Numbers of iterations didn't match expected \
  setting."
  try:
    b.insert(0)
    assert False, "calling insert with only index should raise error."
  except TypeError:
    pass
  try:
    b.insert(1, iterations=10, bias=False)
    assert False, "calling insert with more than one setting should raise error."
  except TypeError:
    pass
  try:
    b.insert('a', learn_rate=0.6)
    assert False, "Should raise TypeError for index is not integer."
  except TypeError:
    pass

# def testSetDefaultFn():
#   """ Testing setting the default activation function."""
#   setting = {
#     "layer_units": [[1,2,1], [10, 5, 5, 10], [2, 2]],
#     "X": np.arange(15).reshape(5,3),
#     "y": np.array([[1,2,3],[4,5,6]])
#   }
#   b = Builder(setting)
  
#   # Using standard default activation function
#   suites = b.build()
#   defaultaf = DEFAULT['af']
#   # First suite
#   info = suites.next()
#   assert info["activationfn"] == ["identity", defaultaf, defaultaf], "Activation functions didn't \
#   match expected setting."
#   # Second suite
#   info = suites.next()
#   assert info["activationfn"] == ["identity", defaultaf, defaultaf, defaultaf], "Activation \
#   functions didn't match expected setting."
#   # Third suite
#   info = suites.next()
#   assert info["activationfn"] == ["identity", defaultaf], "Activation functions didn't match \
#   expected setting."

#   # Using new default activation function
#   af.add("dummyaf", 1)
#   b.setDefaultActivationFn("dimmyaf")
#   suites = b.build()
#   # First suite
#   info = suites.next()
#   assert info["activationfn"] == ["identity", "dummyaf", "dummyaf"], "Activation functions didn't \
#   match expected setting."
#   # Second suite
#   info = suites.next()
#   assert info["activationfn"] == ["identity", "dummyaf", "dummyaf", "dummyaf"], "Activation \
#   functions didn't match expected setting."
#   # Third suite
#   info = suites.next()
#   assert info["activationfn"] == ["identity", "dummyaf"], "Activation functions didn't match \
#   expected setting."

# def testRemove():
#   """ Testing removing value."""
#   b = Builder()
#   b.set(layer_units=[[1,3],[2,3],[3,3],[4,3],[5,3]])
#   assert b.get("layer_units") == [[1,3],[2,3],[3,3],[4,3],[5,3]], "Layer units didn't match \
#   expected setting."
#   assert b.remove("layer_units", 0) == [1,3], "Removed layer units didn't match expected output."
#   assert b.get("layer_units") == [[2,3],[3,3],[4,3],[5,3]], "Layer units didn't match expectet \
#   setting."
#   assert b.remove("layer_units", 2) == [4,3], "Removed layer units didn't match expected output."
#   assert b.get("layer_units") == [[2,3],[3,3],[5,3]], "Layer units didn't match expectet setting."
#   assert b.remove("layer_units", 2) == [5,3], "Removed layer units didn't match expected output."
#   assert b.get("layer_units") == [[2,3],[3,3]], "Layer units didn't match expectet setting."
#   assert b.remove("layer_units") == [3,3], "Removed layer units didn't match expected output."
#   assert b.get("layer_units") == [2,3], "Layer units didn't match expectet setting."
#   assert b.remove("layer_units") == [2,3], "Removed layer units didn't match expected output."
#   assert b.get("layer_units") == [], "Layer units didn't match expectet setting."
#   try:
#     b.remove("layer_units")
#     assert False, "Should raise IndexError for trying to remove when there is no more value."
#   except IndexError:
#     pass
#   try:
#     b.remove(1,1)
#     assert False, "Should raise TypeError for passing non-string key."
#   except TypeError:
#     pass

# def testClear():
#   """ Testing resetting all settings to default."""
#   dummyMinimizer = lambda x: x
#   setting = {
#     "minimizer": dummyMinimizer,
#     "bias": False,
#     "activationfn": ["sigmoid", "identity", "sigmoid"],
#     "iterations": 1337,
#     "learn_rate": 1.0,
#     "layer_units": [10, 20, 1],
#     "errfn": "logitError",
#     "X": np.arange(15).reshape(5,3),
#     "y": np.array([[1,2,3],[4,5,6]])
#   }
#   b = Builder(setting)
#   b.clear()
#   info = b.getSetting()
#   # Settings for the neural network
#   assert info['layer_units'] == DEFAULT['layer_units'], "Layer units didn't match default setting."
#   assert info['activationfn'] == [], "Activation functions didn't match default setting."
#   assert info['bias'] == DEFAULT['bias'], "Bias didn't match default setting."
#   # Settings for the trainning
#   assert info['errfn'] == DEFAULT['errfn'], "Error function didn't match default setting."
#   assert info['learn_rate'] == DEFAULT['learn_rate'], "Learning rate didn't match default setting."
#   assert info['minimizer'] == DEFAULT['minimizer'], "Minimizer didn't match default setting."
#   assert info['iterations'] == DEFAULT['iterations'], "Number of trainning iterations didn't match \
#   default setting."
#   assert info['X'] == [], "Input trainning data (X) should be []."
#   assert info['y'] == [], "Expected output data (y) shoulde be []."

# def testBuild1():
#   """ Testing building a generator of one neural network suite."""
#   dummyMinimizer = lambda x: x
#   setting = {
#     "layer_units": [10, 20, 1],
#     "activationfn": ["sigmoid", "identity", "sigmoid"],
#     "bias": False,
#     "minimizer": dummyMinimizer,
#     "iterations": 1337,
#     "learn_rate": 1.0,
#     "errfn": "logitError",
#     "X": np.arange(15).reshape(5,3),
#     "y": np.array([[1,2,3],[4,5,6]])
#   }
#   b = Builder(setting)
#   suites = b.build()
#   info = suites[0]
#   assert info['layer_units'] == [10,20,1], "Layer units didn't match expected setting."
#   assert info['activationfn'] == ['sigmoid', 'identity', 'sigmoid'], "Activation functions didn't \
#   match expected setting."
#   assert info['bias'] == False, "Bias didn't match expected setting."
#   assert info['minimizer'] == dummyMinimizer, "Minimizer didn't match expected setting."
#   assert info['iterations'] == 1337, "Number of iterations didn't match expected setting."
#   assert info['learn_rate'] == 1.0, "Learning rate didn't match expected setting."
#   assert info['errfn'] == "logitError", "Error function didn't match expected setting."
#   assert np.array_equal(info['X'], np.arange(15).reshape(5,3)), "Input data didn't match expected \
#   setting."
#   assert np.array_equal(info['y'], np.arange(6).reshape(2,3)), "Input data didn't match expected \
#   setting."
#   b.clear("activationfn")
#   suites = b.build()
#   info = suites[0]
#   assert info["activationfn"] == ["identity", "sigmoid", "sigmoid"]

# def testBuild2():
#   """ Testing building a generator of several neural network suite with only one setting of two or
#   more values.
#   """
#   dummyMinimizer = lambda x: x
#   setting = {
#     "layer_units": [[10, 20, 1], [1,2,2,1], [5,2]],
#     "bias": False,
#     "minimizer": dummyMinimizer,
#     "iterations": 1337,
#     "learn_rate": 1.0,
#     "errfn": "logitError",
#     "X": np.arange(15).reshape(5,3),
#     "y": np.array([[1,2,3],[4,5,6]])
#   }
#   b = Builder(setting)

#   suites = b.build()
#   # First suite
#   info = suites[0]
#   assert info['layer_units'] == [10,20,1], "Layer units didn't match expected setting."
#   assert info['activationfn'] == ['identity', 'sigmoid', 'sigmoid'], "Activation functions didn't \
#   match expected setting."
#   assert info['bias'] == False, "Bias didn't match expected setting."
#   assert info['minimizer'] == dummyMinimizer, "Minimizer didn't match expected setting."
#   assert info['iterations'] == 1337, "Number of iterations didn't match expected setting."
#   assert info['learn_rate'] == 1.0, "Learning rate didn't match expected setting."
#   assert info['errfn'] == "logitError", "Error function didn't match expected setting."
#   assert np.array_equal(info['X'], np.arange(15).reshape(5,3)), "Input data didn't match expected \
#   setting."
#   assert np.array_equal(info['y'], np.arange(6).reshape(2,3)), "Input data didn't match expected \
#   setting."
#   # Second suite
#   info = suites[1]
#   assert info['layer_units'] == [1,2,2,1], "Layer units didn't match expected setting."
#   assert info['activationfn'] == ['identity', 'sigmoid', 'sigmoid', 'sigmoid'], "Activation \
#   functions didn't match expected setting."
#   assert info['bias'] == False, "Bias didn't match expected setting."
#   assert info['minimizer'] == dummyMinimizer, "Minimizer didn't match expected setting."
#   assert info['iterations'] == 1337, "Number of iterations didn't match expected setting."
#   assert info['learn_rate'] == 1.0, "Learning rate didn't match expected setting."
#   assert info['errfn'] == "logitError", "Error function didn't match expected setting."
#   assert np.array_equal(info['X'], np.arange(15).reshape(5,3)), "Input data didn't match expected \
#   setting."
#   assert np.array_equal(info['y'], np.arange(6).reshape(2,3)), "Input data didn't match expected \
#   setting."
#   # Third suite
#   info = suites[2]
#   assert info['layer_units'] == [5, 2], "Layer units didn't match expected setting."
#   assert info['activationfn'] == ['identity', 'sigmoid'], "Activation functions didn't match \
#   expected setting."
#   assert info['bias'] == False, "Bias didn't match expected setting."
#   assert info['minimizer'] == dummyMinimizer, "Minimizer didn't match expected setting."
#   assert info['iterations'] == 1337, "Number of iterations didn't match expected setting."
#   assert info['learn_rate'] == 1.0, "Learning rate didn't match expected setting."
#   assert info['errfn'] == "logitError", "Error function didn't match expected setting."
#   assert np.array_equal(info['X'], np.arange(15).reshape(5,3)), "Input data didn't match expected \
#   setting."
#   assert np.array_equal(info['y'], np.arange(6).reshape(2,3)), "Input data didn't match expected \
#   setting."

# def testBuild3():
#   """ Testing building a generator of several neural network suite with more than one setting of two
#   or more values.
#   """
#   dummyMinimizer = lambda x: x
#   setting = {
#     "layer_units": [[10, 20, 1], [1,2,2,1]],
#     "activationfn": [["identity", "sigmoid", "sigmoid"], ["identity", "sigmoid", "sigmoid", 
#     "sigmoid"]],
#     "bias": False,
#     "minimizer": dummyMinimizer,
#     "iterations": [1337, 10],
#     "learn_rate": 1.0,
#     "errfn": ["logitError", "squaredError"],
#     "X": np.arange(15).reshape(5,3),
#     "y": np.array([[1,2,3],[4,5,6]])
#   }
#   b = Builder(setting)

#   suites = b.build()
#   # First suite
#   info = suites[0]
#   assert info['layer_units'] == [10,20,1], "Layer units didn't match expected setting."
#   assert info['activationfn'] == ['identity', 'sigmoid', 'sigmoid'], "Activation functions didn't \
#   match expected setting."
#   assert info['bias'] == False, "Bias didn't match expected setting."
#   assert info['minimizer'] == dummyMinimizer, "Minimizer didn't match expected setting."
#   assert info['iterations'] == 1337, "Number of iterations didn't match expected setting."
#   assert info['learn_rate'] == 1.0, "Learning rate didn't match expected setting."
#   assert info['errfn'] == "logitError", "Error function didn't match expected setting."
#   assert np.array_equal(info['X'], np.arange(15).reshape(5,3)), "Input data didn't match expected \
#   setting."
#   assert np.array_equal(info['y'], np.arange(6).reshape(2,3)), "Input data didn't match expected \
#   setting."
#   # Second suite
#   info = suites[1]
#   assert info['layer_units'] == [1,2,2,1], "Layer units didn't match expected setting."
#   assert info['activationfn'] == ['identity', 'sigmoid', 'sigmoid', 'sigmoid'], "Activation \
#   functions didn't match expected setting."
#   assert info['bias'] == False, "Bias didn't match expected setting."
#   assert info['minimizer'] == dummyMinimizer, "Minimizer didn't match expected setting."
#   assert info['iterations'] == 10, "Number of iterations didn't match expected setting."
#   assert info['learn_rate'] == 1.0, "Learning rate didn't match expected setting."
#   assert info['errfn'] == "squaredError", "Error function didn't match expected setting."
#   assert np.array_equal(info['X'], np.arange(15).reshape(5,3)), "Input data didn't match expected \
#   setting."
#   assert np.array_equal(info['y'], np.arange(6).reshape(2,3)), "Input data didn't match expected \
#   setting."

#   b.clear("activationfn", "layer_units")
#   suites = b.build()
#   # First suite
#   info = suites[0]
#   assert info['layer_units'] == DEFAULT["layer_units"], "Layer units didn't match expected setting."
#   assert info['activationfn'] == ['identity', 'sigmoid'], "Activation functions didn't match \
#   expected setting."
#   assert info['bias'] == False, "Bias didn't match expected setting."
#   assert info['minimizer'] == dummyMinimizer, "Minimizer didn't match expected setting."
#   assert info['iterations'] == 1337, "Number of iterations didn't match expected setting."
#   assert info['learn_rate'] == 1.0, "Learning rate didn't match expected setting."
#   assert info['errfn'] == "logitError", "Error function didn't match expected setting."
#   assert np.array_equal(info['X'], np.arange(15).reshape(5,3)), "Input data didn't match expected \
#   setting."
#   assert np.array_equal(info['y'], np.arange(6).reshape(2,3)), "Input data didn't match expected \
#   setting."
#   # Second suite
#   info = suites[1]
#   assert info['layer_units'] == DEFAULT["layer_units"], "Layer units didn't match expected setting."
#   assert info['activationfn'] == ['identity', 'sigmoid'], "Activation functions didn't match \
#   expected setting."
#   assert info['bias'] == False, "Bias didn't match expected setting."
#   assert info['minimizer'] == dummyMinimizer, "Minimizer didn't match expected setting."
#   assert info['iterations'] == 10, "Number of iterations didn't match expected setting."
#   assert info['learn_rate'] == 1.0, "Learning rate didn't match expected setting."
#   assert info['errfn'] == "squaredError", "Error function didn't match expected setting."
#   assert np.array_equal(info['X'], np.arange(15).reshape(5,3)), "Input data didn't match expected \
#   setting."
#   assert np.array_equal(info['y'], np.arange(6).reshape(2,3)), "Input data didn't match expected \
#   setting."  

# def testBuildErrors():
#   """ Testing the errors of build."""
#   try:
#     b = Builder()
#     suites = b.build()
#     assert False, "Should raise ValueError for not setting input and output trainning data."
#   except ValueError:
#     pass

#   dummyMinimizer = lambda x: x
#   setting = {
#     "layer_units": [10, 20, 1],
#     "activationfn": ["sigmoid", "sigmoid"],
#     "bias": False,
#     "minimizer": dummyMinimizer,
#     "iterations": 1337,
#     "learn_rate": 1.0,
#     "errfn": "logitError",
#     "X": np.arange(15).reshape(5,3),
#     "y": np.array([[1,2,3],[4,5,6]])
#   }
#   b = Builder(setting)
#   try:
#     suites = b.build()
#     assert False, "Should raise ValueError for number of activation functions didn't match number \
#     of layers."
#   except ValueError:
#     pass

#   b.clear("activationfn")
#   b.set(bias=[True, False], learn_rate=[0.5, 0, 0.96])
#   try:
#     suites = b.build()
#     assert False, "Should raise ValueError for number of values for bias and learning rate didn't \
#     match."
#   except ValueError:
#     pass  

#   b.set(bias=True, learn_rate=[0.5, 1.0])
#   b.set(activationfn=[["identity", "sigmoid", "sigmoid"], ["identity", "sigmoid"]])
#   try:
#     suites = b.build()
#     assert False, "Should raise ValueError for in the second suite, number of activation functions \
#     didn't match number of layers. " 
#   except ValueError:
#     pass

#   b.append(layer_units=[10,5,5])
#   try:
#     suites = b.build()
#     assert False, "Should raise ValueError for in the second suite, number of activation functions \
#     didn't match number of layers. " 
#   except ValueError:
#     pass
