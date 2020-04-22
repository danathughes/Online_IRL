import numpy as np
import scipy.sparse
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix, dok_matrix


class AbstractTransition:
	 """
	 An abstract representation of transitions (i.e., dynamics) of an
	 environment.
	 """

	 def __init__(self):
			"""
			Create a new transition function.
			"""

			pass


	 def __call__(self, state, action, next_state):
			"""
			Return the probability of transitioning to next_state, given state and
			action.
			"""

			raise NotImplementedError

	 ## TODO: Figure out which other parameters should be set here









class DiscreteTransition(AbstractTransition):
	 """
	 DiscreteTransition represents the transition function as a 3D tensor of the
	 form (state x action x next_state).
	 """

	 def __init__(self, num_states, num_actions, initial_value = 0):
			"""
			Create a discrete transition function for the provided state and action
			space dimensionality.

			num_states    - the number of states in the transition model
			num_actions   - the number of actions in the transition model
			initial_value - (optional) the value to initialize each probability to
											Default value is 0.
			"""

			AbstractTransition.__init__(self)

			self.num_states = num_states
			self.num_actions = num_actions

			# Represent the transition function internally using a numpy array.  
			# Initialize the array to the provided initial value.
			self.__transition = np.ones((self.num_states, self.num_actions, self.num_states))
			self.__transition *= initial_value


	 def set(self, state, action, next_state, probability):
			"""
			Set the probability of the transition provided.  

			Multiple transitions can be set simulataneously by passing iterable
			objects for the state, action, next_state, and probability arguments.
			In the event that 
			"""

			self.__transition[state, action, next_state] = probability


	 def __call__(self, state, action, next_state):
			"""
			Return the probability of transitioning to next_state from state when
			action is performed
			"""

			# TODO: Validate that state, action, and next state are within range

			return self.__transition[state,action,next_state]


	 def __getitem__(self, index):
			"""
			Returns the probability of transitioning to next_state from state when
			action is performed.  Allows for slicing among multiple dimensions.
			"""

			# TODO: Validate that state, action, and next state are within range

			return self.__transition[index]


	 def __setitem__(self, index, value):
			"""
			Sets the transition probabilit(ies) at the index to the provided value
			"""

			self.__transition[index] = value


	 def sample(self, state, action, shape = None):
			"""
			Generate a sample of the next_state given the state / action pair.

			state  - current state
			action - action performed
			shape (optional) - the number of samples to produce.  If 'None',
												 produce a single sample, otherwise, produce a
												 numpy array of samples in the provided shape.
			"""

			# TODO: Validate that state, action, and next state are within range

			# What's the distribution over the next_state
			state_distribution = self[state, action,:]

			# Pick a state given the distribution.  If a shape is provided, create
			# a set of samples conforming to the shape
			if shape is None:
				 return np.random.choice(range(self.num_states), p=state_distribution)
			else:
				 return np.random.choice(range(self.num_states), size=shape, p=state_distribution)


	 def asArray(self):
			"""
			Return a numpy array representing the transition tensor P(s,a,s')
			"""

			return self.__transition
			















class SparseTransition(AbstractTransition):
	"""
	SparseTransition represents the transition function as a list of sparse 
	matricies.  Each matrix represents the state to nextState transitions for a
	specific action.
	"""

	def __init__(self, num_states, num_actions):
		"""
		Create a discrete transition function for the provided state and action
		space dimensionality using a set of sparse matricies.

		num_states    - the number of states in the transition model
		num_actions   - the number of actions in the transition model
		"""

		AbstractTransition.__init__(self)

		self.num_states = num_states
		self.num_actions = num_actions

		# Represent the transition function internally using a list of sparse
		# matrices.  List is indexed by action
		self.__transition = [dok_matrix((self.num_states, self.num_states)) for _ in range(self.num_actions)]


	def set(self, state, action, next_state, probability):
		"""
		Set the probability of the transition provided.
		"""

		self.__transition[action][state, next_state] = probability


	def __call__(self, state, action, next_state):
		"""
		Return the probability of transitioning to next_state from state when
		action is performed
		"""

		# TODO: Validate that state, action, and next state are within range

		return self.__transition[action][state,next_state]


	def __getitem__(self, index):
		"""
		Returns the probability of transitioning to next_state from state when
		action is performed.  Allows for slicing among multiple dimensions.
		"""

		# TODO: Validate that state, action, and next state are within range
		state,action,nextState = index


		return self.__transition[action][state,nextState]


	def __setitem__(self, index, value):
		"""
		Sets the transition probabilit(ies) at the index to the provided value
		"""

		state,action,nextState = index

		self.__transition[action][state,nextState] = value


	def sample(self, state, action, shape = None):
		"""
		Generate a sample of the next_state given the state / action pair.

		state  - current state
		action - action performed
		shape (optional) - the number of samples to produce.  If 'None',
											 produce a single sample, otherwise, produce a
											 numpy array of samples in the provided shape.
		"""

		# TODO: Validate that state, action, and next state are within range

		# What's the distribution over the next_state
		state_distribution = self[state, action,:]

		state_distribution = np.array(state_distribution.todense()).flatten()

		# Pick a state given the distribution.  If a shape is provided, create
		# a set of samples conforming to the shape
		if shape is None:
			return np.random.choice(range(self.num_states), p=state_distribution)
		else:
			return np.random.choice(range(self.num_states), size=shape, p=state_distribution)


	def finalize(self):
		"""
		Convert the matrix to a format more suitable for tensor operations
		"""

		self.__transition = [m.tocsr() for m in self.__transition]


	def asArray(self):
		"""
		Return a numpy array representing the transition tensor P(s,a,s')
		"""

		return self.__transition
