import numpy as np
from pymind.util import assertType

def testAssertType1():
  assertType("function1", "test", 3, int)
  np.testing.assert_raises(TypeError, assertType, "function2", "var", 2, str)
