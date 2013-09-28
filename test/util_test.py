import numpy as np
import pymind.util as pyu

def testGenerateMatrix():
  """ Tests if generate_matrix produces correct matrix. """
  mat = pyu.generate_matrix(10, 10, np.cos)

  assert type(mat) is np.matrix, "generate_matrix should return a matrix."
  assert mat.size == 100, "generate_matrix should have 100 elements."
  np.testing.assert_approx_equal(mat.item(40), -.666938062, significant=6,
    err_msg = "generate_matrix should calculate cos(40) and store it in at index 40.")

def testUnrollReshape():
  """ Tests if unroll_matrices and reshape_vector unrolls/reshapes correctly. """
  matrices = (pyu.generate_matrix(8, 11), pyu.generate_matrix(6, 9))

  unrolled_vector = pyu.unroll_matrices(matrices)
  reshaped_matrices = pyu.reshape_vector(unrolled_vector, ((8, 11), (6, 9)))

  for i in xrange(len(reshaped_matrices)):
    np.testing.assert_array_equal(reshaped_matrices[i], matrices[i],
      err_msg = "The reshapedWeight at index %d is not the same as the \
        same weight in the neural network." % i)
