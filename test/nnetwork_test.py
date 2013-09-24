import numpy as np
from pymind.components import NNLayer, NeuralNetwork
from pymind.activationfn import *

# Utility method for testing purposes only
def setWeight(network, index, weight):
  network.weights[index] = weight

def testNNetworkConstruction1():
  """ Test construction of a neural network with a single hidden layer. """
  params = {
    "input_units": 5,
    "output_units": 3,
    "hidden_units": 4,
    "activationfn": [identity, sigmoid, sigmoid],
    "bias": True
  }
  nnet = NeuralNetwork(params)
  input_layer = nnet.layers[0]

  assert len(nnet.layers) == 3, "There should be three neural network layers."
  assert type(input_layer) == NNLayer, "The first neural network layer should be of type NNLayer."
  assert input_layer.num_input == params["input_units"], \
    "The first neural network layer should have %d input_units: %d" \
    % (params["input_units"], input_layer.num_input)
  assert nnet.weights[0].shape == (params["hidden_units"], params["input_units"] + 1), \
    "The first set of weights in the neural network should have a shape %s" \
    % (params["hidden_units"], (params["input_units"] + 1))

def testNNetworkConstruction2():
  """ Test construction of a neural network with multiple hidden layers. """
  params = {
    "input_units": 5,
    "output_units": 3,
    "hidden_units": [4, 4, 3],
    "activationfn": [identity, sigmoid, sigmoid, sigmoid, sigmoid],
    "bias": False
  }
  nnet = NeuralNetwork(params)
  input_layer = nnet.layers[0]

  assert len(nnet.layers) == 5, "There should be five neural network layers."
  assert type(input_layer) == NNLayer, "The first neural network layer should be of type nnlayer."
  assert input_layer.num_input == params["input_units"], \
    "The first neural network layer should have %d input_units." \
    % params["input_units"]
  assert nnet.weights[0].shape == (params["hidden_units"][0], params["input_units"]), \
    "The first set of weights in the neural network should have a shape %s." \
    % (params["hidden_units"][0], params["input_units"])

def testForwardProp1():
  """ Test forward propogation for a simple neural network. """
  params = {
    "input_units": 2,
    "output_units": 1,
    "hidden_units": 2,
    "activationfn": [identity, identity, identity],
    "bias": True
  }
  nnet = NeuralNetwork(params)

  # Manually set weights for testing purposes
  weight0 = np.matrix(np.ones(6)).reshape(2, 3)
  weight1 = np.matrix(np.ones(3)).reshape(1, 3)
  setWeight(nnet, 0, weight0)
  setWeight(nnet, 1, weight1)

  x = np.matrix([1, 1]).T
  z, a = nnet.feedForward(x)

  np.testing.assert_array_equal(a[1], np.matrix([1, 3, 3]).T,
    err_msg = "The hidden layer activation values should be a column vector with elements 1, 3, 3: %r" % a[1])
  assert a[2].size == 1, "The output should be a one element matrix: %r" % a[2].size
  assert a[2].item(0) == 7, "The output value should be 7: %d" % a[2].item(0)

def testForwardProp2():
  """ Test forward propogation for a complex neural network. """
  params = {
    "input_units": 3,
    "hidden_units": [2, 2],
    "output_units": 1,
    "activationfn": [identity, sigmoid, identity, sigmoid],
    "bias": True
  }
  nnet = NeuralNetwork(params)

  # Manually set weights for testing purposes
  weight0 = np.matrix(np.ones(8)).reshape(2, 4)
  weight1 = np.matrix(np.ones(6)).reshape(2, 3)
  weight2 = np.matrix(np.ones(3)).reshape(1, 3)


  setWeight(nnet, 0, weight0)
  setWeight(nnet, 1, weight1)
  setWeight(nnet, 2, weight2)

  # Feed forward
  x = np.matrix(np.ones(6)).reshape(3, 2)
  x[:, 1] = 2
  z, a = nnet.feedForward(x)

  # Create actual result (pre-calculated)
  z_test = list()
  a_test = list()

  z_test.append(x)
  a_test.append(np.matrix(np.vstack((np.ones((1, 2)), x))))
  z_test.append(np.matrix([4, 7, 4, 7]).reshape(2, 2))
  a_test.append(np.matrix([1, 1, 0.98201, 0.99909, 0.98201, 0.99909]).reshape(3, 2))
  z_test.append(np.matrix([2.9640, 2.9982, 2.9640, 2.9982]).reshape(2, 2))
  a_test.append(np.matrix([1, 1, 2.9640, 2.9982, 2.9640, 2.9982]).reshape(3, 2))
  z_test.append(np.matrix([6.9281, 6.9964]))
  a_test.append(np.matrix([0.99902, 0.99909]))

  for i in xrange(len(z_test)):
    np.testing.assert_array_almost_equal(z_test[i], z[i], decimal = 4,
      err_msg = "The output z at index %d is incorrect" % i)
    np.testing.assert_array_almost_equal(a_test[i], a[i], decimal = 4,
      err_msg = "The output a at index %d is incorrect" % i)

def testForwardProp3():
  """ Test forward propogation for high accuracy. """
  params = {
    "input_units": 5,
    "output_units": 2,
    "hidden_units": 4,
    "activationfn": [identity, sigmoid, sigmoid],
    "bias": True
  }
  nnet = NeuralNetwork(params)

  # Manually set weights for testing purposes
  weight0 = np.matrix(np.sin(np.arange(24)).reshape(4, 6))
  weight1 = np.matrix(np.sin(np.arange(10)).reshape(2, 5))

  setWeight(nnet, 0, weight0)
  setWeight(nnet, 1, weight1)

  # Create input vector x
  x = np.matrix(np.sin(np.arange(5))).reshape(5, 1)

  # Feed forward
  z, a = nnet.feedForward(x)

  # Create actual result (pre-calculated)
  zt = list()
  at = list()

  zt.append(np.matrix([0.000000000000000, 0.841470984807897, 0.909297426825682,
    0.141120008059867,-0.756802495307928]).T)
  at.append(np.matrix([1.000000000000000, 0.000000000000000, 0.841470984807897,
    0.909297426825682, 0.141120008059867,-0.756802495307928]).T)
  zt.append(np.matrix([1.51238377107558, 1.60786185814031, 1.57525859137397, 1.41717112831568]).T)
  at.append(np.matrix([1.000000000000000, 0.819414212940607, 0.833114321891000,
    0.828531970380330, 0.804894555480342]).T)
  zt.append(np.matrix([0.954838224166471, 0.510890501950654]).T)
  at.append(np.matrix([0.722087142278112, 0.625015205701210]).T)

  # Comparison!

  for i in xrange(len(z)):
    np.testing.assert_array_almost_equal(z[i], zt[i], decimal=10,
      err_msg = "The output z at index %d is incorrect" % i)
    np.testing.assert_array_almost_equal(a[i], at[i], decimal = 10,
      err_msg = "The output a at index %d is incorrect" % i)

