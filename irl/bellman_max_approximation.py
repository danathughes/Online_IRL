## bellman_approximation.py
##
## Approximations to the maximum function used in Bellman iteration

import numpy as np

class PNorm:
   """
   """

   def __init__(self, k, EPSILON=1e-8):
      """
      """

      self.k = k
      self.EPSILON = EPSILON

   def __call__(self, values):
      """
      """

      total = np.sum(values**self.k, axis=1)

      return np.power(total, 1.0/self.k)


   def gradient(self, values):
      """
      """

      val1 = np.sum(values**self.k, axis=1)
      exponent = (1.0 - self.k)/self.k

      # Having a 0 as the value used in the power function seems to cause some trouble, add a small epsilon if needed
      val1[val1==0] = self.EPSILON

      C = np.power(val1, exponent) / self.k

      val2 = self.k*values**(self.k-1)

      return C[:,None]*val2


class Softmax:
   """
   """

   def __init__(self, k, MAX_THRESHOLD=1.0e300):
      """
      """

      self.k = k
      self.MAX_THRESHOLD = MAX_THRESHOLD


   def __call__(self, values):
      """
      """

      softmax = np.exp(self.k*values)

      # Make sure that all values are an actual number -- easy to add up to a non
      softmax[softmax_values < -self.MAX_THRESHOLD] = -self.MAX_THRESHOLD
      softmax[softmax_values > self.MAX_THRESHOLD] = self.MAX_THRESHOLD

      total = np.sum(softmax, axis=1)

      return np.log(total) / self.k


   def gradient(self, values):
      """
      """

      grad = np.exp(self.k*values)

      # Make sure that all values are an actual number -- easy to add up to a non
      grad[grad < -self.MAX_THRESHOLD] = -self.MAX_THRESHOLD
      grad[grad > self.MAX_THRESHOLD] = self.MAX_THRESHOLD

      Z = np.sum(grad, axis=1)

      return grad / Z[:,None]

