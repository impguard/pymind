import numpy as np

def initRandParams(num_input, num_output):
  eps_init = np.sqrt(6) / np.sqrt(num_input + num_output)
  return np.matrix(np.random.rand(num_output, num_input) * 2 * eps_init - eps_init)
