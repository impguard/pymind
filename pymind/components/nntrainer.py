from nnetwork import NeuralNetwork
from nnlayer import NNLayer

class NNTrainer(object):
  def __init__(self, nn, err_fn):
    self.nn = nn
    self.err_fn = err_fn

  def train(self, data):
    self.nn
