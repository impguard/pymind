import numpy as np
from pymind.components import NNetworkLayer, NeuralNetwork
from pymind.matrixfunc import *

# Utility method for testing purposes only
def setWeight(network, index, weight):
  network.weights[index] = weight

def testNNetworkConstruction():
  # Test single hidden layer
  params = {
    "input_units": 5,
    "output_units": 3,
    "hidden_units": 4,
    "activation_fn": [identity, sigmoid, sigmoid],
    "bias": True
  }
  nnet = NeuralNetwork(params)
  input_layer = nnet.layers[0]

  assert len(nnet.layers) == 3, "There should be three neural network layers: %d" % len(nnet.layers)
  assert type(input_layer) == NNetworkLayer, \
    "The first neural network layer should be of type NNetworkLayer: %s" % type(input_layer)
  assert input_layer.num_input == params["input_units"], \
    "The first neural network layer should have %d input_units: %d" % (params["input_units"], input_layer.num_input)
  assert nnet.weights[0].shape == (params["hidden_units"], params["input_units"] + 1), \
    "The first set of weights in the neural network should have a shape %s: %s" \
    % (params["hidden_units"], (params["input_units"] + 1), nnet.weights[0].shape)

  # Test multiple hidden layers
  params = {
    "input_units": 5,
    "output_units": 3,
    "hidden_units": [4, 4, 3],
    "activation_fn": [identity, sigmoid, sigmoid, sigmoid, sigmoid],
    "bias": False
  }
  nnet = NeuralNetwork(params)
  input_layer = nnet.layers[0]

  assert len(nnet.layers) == 5, "There should be five neural network layers: %d" % len(nnet.layers)
  assert type(input_layer) == NNetworkLayer, \
    "The first neural network layer should be of type nnlayer: %s" % type(input_layer)
  assert input_layer.num_input == params["input_units"], \
    "The first neural network layer should have %d input_units: %d" % (params["input_units"], input_layer.num_input)
  assert nnet.weights[0].shape == (params["hidden_units"][0], params["input_units"]), \
    "The first set of weights in the neural network should have a shape %s: %s" \
    % ((params["hidden_units"][0], params["input_units"]), nnet.weights[0].shape)

def testForwardProp():
  # Create a simple neural network
  params = {
    "input_units": 2,
    "output_units": 1,
    "hidden_units": 2,
    "activation_fn": [identity, identity, identity],
    "bias": True
  }
  nnet = NeuralNetwork(params)

  # Manually set weights for testing purposes
  weight0 = np.matrix(np.ones(6)).reshape(2, 3)
  weight1 = np.matrix(np.ones(3)).reshape(1, 3)
  setWeight(nnet, 0, weight0)
  setWeight(nnet, 1, weight1)

  x = np.matrix([1, 1]).T
  z, a = nnet.feed_forward(x)

  np.testing.assert_array_equal(a[1], np.matrix([1, 3, 3]).T,
    err_msg = "The hidden layer activation values should be a column vector with elements 1, 3, 3: %r" % a[1])
  assert a[2].size == 1, "The output should be a one element matrix: %r" % a[2].size
  assert a[2].item(0) == 7, "The output value should be 7: %d" % a[2].item(0)

  # Create a more complex neural network
  params = {
    "input_units": 3,
    "hidden_units": [2, 2],
    "output_units": 1,
    "activation_fn": [identity, sigmoid, identity, sigmoid],
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

  x = np.matrix(np.ones(6)).reshape(3, 2)
  x[:, 1] = 2
  z, a = nnet.feed_forward(x)

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

  for i in range(len(z_test)):
    np.testing.assert_array_almost_equal(z_test[i], z[i], decimal = 4,
      err_msg = "The output z at index %d should be \n %r \n != %r" % (i, z_test[i], z[i]))
    np.testing.assert_array_almost_equal(a_test[i], a[i], decimal = 4,
      err_msg = "The output a at index %d should be \n %r \n != %r" % (i, a_test[i], a[i]))
