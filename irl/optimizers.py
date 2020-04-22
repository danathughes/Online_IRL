import numpy as np 


class GradientAscentOptimizer:
	"""
	Simple gradient ascent optimizer

	GradientDescentOptimizer is initialized with a reference to the parameter
	to optimize.  Subsequent calls with 
	"""

	def __init__(self, learning_rate,
				 difference_threshold=0.001,
				 min_iterations=10,
				 max_iterations=100,
				 max_value = 2.0,
				 min_value = -2.0):
		"""
		Create a Gradient Descent Optimizer

		learning_rate - learning rate of the optimizer
		difference_threshold - maximum difference between subsequent iterations
		                       before stopping
		min_iterations - minimum number of iterations to perform before stopping
		max_iterations - maximum number of iterations allowed
		"""

		self.learning_rate = learning_rate

		self.difference_threshold = difference_threshold
		self.min_iterations = min_iterations
		self.max_iterations = max_iterations

		self.max_value = max_value
		self.min_value = min_value

		self.parameters = None
		self.old_parameters = None
		self.iter_num = 0

	def initialize(self, parameters):
		"""
		Initialize gradient descent on the provided parametres
		"""

		self.parameters = parameters
		self.old_parameters = None
		self.iter_num = 0


	def update(self, gradient):
		"""
		Update the parameters with the gradient
		"""

		# Save a copy of the old parameters
		self.old_parameters = self.parameters.copy()
		self.parameters += self.learning_rate * gradient

		self.parameters[self.parameters > self.max_value] = self.max_value
		self.parameters[self.parameters < self.min_value] = self.min_value

		self.iter_num += 1


	def done(self):
		"""
		Indicate whether stopping criteria has been reached
		"""

		# Need to have at least computed something!
		if self.old_parameters is None:
			return False

		# Need to have at least performed the minimum number of iterations
		if self.iter_num < self.min_iterations:
			return False

		# Stop if the maximum number of iterations has been reached
		if self.iter_num > self.max_iterations:
			return True

		# Stop if the absolute difference between all parameters is less than
		# the threshold
		diff = np.abs(self.parameters - self.old_parameters)
		return np.all(diff < self.difference_threshold)