from nnetwork import NeuralNetwork
from nnlayer import NNLayer
from collections import deque

class NNTrainer(object):
  def __init__(self, nn, err_fn, learn_rate):
    self.nn = nn
    self.err_fn = err_fn
    self.learn_rate = learn_rate

  def train(self, X, y):

    # Minimize

  def createCostFn(self, X, y)
    m = X.shape[1]
    ext_y = np.tile(y, (1, m))

    # Part 1: Feed-forward + Get error
    z, a = self.nn.feed_forward(X)
    cost = self.err_fn.calc(a[-1], ext_y)

    # Part 2: Backpropogation
    d = deque()
    lastD = np.multiply(self.nn.activationfn[-1].grad(z[-1]) , self.err_fn.grad(a[-1], ext_y)
    d.appendleft(lastD)

    for i in range(len(self.nn.activationfn) - 2, 1, -1):
      fn = self.nn.activationfn[i]
      nextD = np.multiply(self.nn.weights[i][:, 1:].T * d[0], fn.grad(z[i]))
      d.appendleft(nextD)

    d.appendleft(None) # Filler so the indexes matchup

    # Get gradient
