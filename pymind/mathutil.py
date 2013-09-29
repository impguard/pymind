""" A set of math utility functions. """

import numpy as np
from scipy.optimize import minimize

def reshape_vector(unrolled_vector, dimensions):
  """ Reshapes an unrolled column or row vector of weights into the proper sizes.

  Assumes that the correct number of weights are passed, and uses a list of dimensions to determine
  how to reshape the unrolled weights. Will return a view of the original column or row vector.

  Arguments:
  unrolled_vector -- A column or row vector of unrolled weights
  dimensions -- A list of tuples representing the desired dimensions for each reshaped matrix
  Returns:
  A list of matrices for each reshaped matrix.

  Note: This method assumes that the size of unrolled_vector is correct.
  """
  try:
    unrolled_vector = unrolled_vector if unrolled_vector.shape[1] == 1 else unrolled_vector.T
    matrices = list()
    curr_index = 0
    for dimension in dimensions:
      shape = dimension
      size = reduce(lambda t, i: t * i, dimension)
      unrolled_chunk = unrolled_vector[curr_index:curr_index + size]
      matrix = unrolled_chunk.reshape(shape)
      matrices.append(matrix)
      curr_index += size
    return matrices
  except ValueError:
    print "Reshaping matrices failed. Most likely due to incorrect size of unrolled_vector."
    raise

def unroll_matrices(matrices):
  """ Unrolls a list of matrices into a column vector.

  Does not return a view of the original matrices, makes a copy.

  Returns:
  A column vector representing the unrolled matrices.
  """
  unrolled_list = [matrix.ravel() for matrix in matrices]
  return np.hstack(unrolled_list).T

def generate_matrix(rows, columns, fn = np.sin):
  """ Generates a matrix of a given shape using the passed function.

  Uses numpy.arange to create a row vector from 0 to the desired size, then reshapes the matrix to
  the desired dimensions. Finally, applies the supplied numpy universal function to the matrix,
  which by default is the sine function.

  Arguments:
  rows -- the number of desired rows
  columns -- the number of desired columns
  fn -- a numpy universal function (default: numpy.sin)
  Returns
  A matrix with dimensions (rows x columns) mutated with fn.
  """
  size = rows * columns
  matrix = np.arange(size).reshape(rows, columns)
  return fn(np.matrix(matrix))

def create_minimizer(method = 'BFGS', tolerance = 1e-3, iterations = 50, display = False):
  """ Creates a minimizer using scipy.optimize optimization functions. """
  def minimizer(fn, initial_guess):
    return minimize(fn, initial_guess,
      method = method,
      jac = True,
      tol = tolerance,
      options = {
        'maxiter': iterations,
        'disp': display
      })
  return minimizer
