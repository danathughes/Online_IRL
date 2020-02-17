## policy.py
##
## A set of classes defining a policy for agents to perform within an MDP.

import numpy as np

class AbstractPolicy:
	"""
	An abstract policy defines methods policies will implement.
	"""

	def __init__(self):
		"""
		AbstractPolicy does no initialization
		"""

		pass


	def __getitem__(self, index):
		"""
		Return the probability for the given state / action pair
		
		index - a (state,action) tuple

		return - the probability of performing the action in the state
		"""

		raise NotImplementedError


	def __setitem__(self, index, value):
		"""
		Set the probability of the given state / action pair

		index - a (state,action) tuple
		value - the probability of performing the action in the state
		"""

		raise NotImplementedError


	def getActionDistribution(self, state):
		"""
		Return a distribution over actions for the given state

		state  - the state (or array-like) to calculate action distributions over

		return - a probability distribution (or array-like)
		"""

		raise NotImplementedError


	def getActionProbability(self, state, action):
		"""
		Return the probability of performing the given action in the given 
		state

		state  - the state (or array-like) to use to select the action
		action - the action (or array-like) to be performed in the state

		return - a probability (or array-like) 
		"""

		raise NotImplementedError


	def selectAction(self, state):
		"""
		Select an action for the given state

		state - the state to use to select the action
		"""

		raise NotImplementedError


	def likelihood(self, trajectory):
		"""
		Get the likelihood of the trajectory under the provided policy

		trajectory - a sequence of state / action pairs

		return - a log-likelihood score of the trajectory under the policy
		"""

		raise NotImplementedError


	def clone(self):
		"""
		Create a clone of the policy

		return - a clone of the policy
		"""

		raise NotImplementedError



class DiscreteDeterministicPolicy(AbstractPolicy):
	"""
	A discrete deterministic policy generates a single action for any given 
	state with probability of one.
	"""

	def __init__(self, stateSpace, actionSpace):
		"""
		Create a deterministic policy that maps states to actions
		"""

		self.stateSpace = stateSpace
		self.actionSpace = actionSpace

		# The policy can be represented as mapping states in the state space to
		# actions.  Initialize the policy to take a random action in each state
		self.policy_map = { state: random.choice(self.actionSpace) for state in self.stateSpace }


	def __getitem__(self, state):
		"""
		Return the probability for the given state / action pair
		
		state - a state

		return - the action to perform in the state
		"""

		return self.policy_map[state]


	def __setitem__(self, state, action):
		"""
		Set the action to perform in the passed state

		state - a state
		action - the action to perform in the state
		"""

		self.policy_map[state] = action


	def getActionDistribution(self, state):
		"""
		Return a distribution over actions for the given state.  Since the 
		policy is deterministic, it returns a one-hot vector indicating the
		action selected.

		state  - the state (or array-like) to calculate action distributions over

		return - a probability distribution (or array-like)
		"""

		# The policy is deterministic, so most entries are zero
		distribution = np.zeros((len(self.actionSpace),))

		# Set the probability of the action to 1.0
		actionNum = self.actionSpace(self.policy_map[state])
		distribution[actionNum] = 1.0

		return distribution


	def getActionProbability(self, state, action):
		"""
		Return the probability of performing the given action in the given 
		state.  Returns 1.0 if the action is the one to perform in the state,
		otherwise returns zero.

		state  - the state (or array-like) to use to select the action
		action - the action (or array-like) to be performed in the state

		return - a probability (or array-like) 
		"""

		return 1.0 if self.policy_map[state] == action else 0.0


	def selectAction(self, state):
		"""
		Select an action for the given state

		state - the state to use to select the action
		"""

		return self.policy_map[state]


	def likelihood(self, trajectory):
		"""
		Get the log-likelihood of the trajectory under the provided policy

		trajectory - a sequence of state / action pairs

		return - 0.0 if the trajectory follows the policy, -inf otherwise
		"""

		likelihood = 0.0

		for state, action in trajectory:
			likelihood += -np.inf if self.getActionProbability(state, action) == 0

		return likelihood


	def clone(self):
		"""
		Create a clone of the policy

		return - a clone of the policy
		"""

		# Create a policy with the same state and action space, and copy over
		# the policy map.
		policy = DiscreteDeterministicPolicy(self.stateSpace, self.actionSpace)
		for state, action in self.policy_map.items():
			policy[state] = action

		return policy




class DiscreteStochasticPolicy(AbstractPolicy):
	"""
	A discrete deterministic policy generates a single action for any given 
	state with probability of one.
	"""

	def __init__(self, stateSpace, actionSpace):
		"""
		Create a deterministic policy that maps states to actions
		"""

		self.stateSpace = stateSpace
		self.actionSpace = actionSpace

		# The policy can be represented as a 2D array.  Initialize to a random
		# distribution for each state.
		self.policy = np.random.uniform(0.0, 1.0, (len(self.stateSpace), len(self.actionSpace)))

		# Calculate the partition function for each row, and divide each row by
		# the partition function
		Z = np.sum(self.policy, axis=1)
		self.policy /= np.vstack([Z]*len(self.actionSpace)).T


	def __getitem__(self, index):
		"""
		Return the probability for the given state / action pair
		
		index - a (state,action) tuple

		return - the action to perform in the state
		"""


		state, action = index

		return self.policy[self.stateSpace(state), self.actionSpace(action)]


	def __setitem__(self, index, probability):
		"""
		Set the action to perform in the passed state

		index - a (state,action) tuple
		probability - the probability of performing the action in the state
		"""

		state, action = index

		self.policy_map[self.stateSpace(state), self.actionSpace(action)] = prob


	def getActionDistribution(self, state):
		"""
		Return a distribution over actions for the given state.  Since the 
		policy is deterministic, it returns a one-hot vector indicating the
		action selected.

		state  - the state (or array-like) to calculate action distributions over

		return - a probability distribution (or array-like)
		"""

		return self.policy[self.stateSpace(state),:]


	def getActionProbability(self, state, action):
		"""
		Return the probability of performing the given action in the given 
		state.  Returns 1.0 if the action is the one to perform in the state,
		otherwise returns zero.

		state  - the state (or array-like) to use to select the action
		action - the action (or array-like) to be performed in the state

		return - a probability (or array-like) 
		"""

		return self[state,action]


	def selectAction(self, state):
		"""
		Select an action for the given state

		state - the state to use to select the action
		"""

		# Get the action distribution
		actionDistribution = self.getActionDistribution(state)

		# Select an index based on the distribution
		actionNum = np.random.choice(len(self.actionSpace), p=actionDistribution)

		return self.actionSpace[actionNum]


	def likelihood(self, trajectory, eps=1e-10):
		"""
		Get the log-likelihood of the trajectory under the provided policy

		trajectory - a sequence of state / action pairs
		eps - (optional) small value to use in place of entries of zero, to
		      avoid NaN results from the log function. Default = 1e-10

		return - the log-likelihood of the trajectory under the policy
		"""

		likelihood = 0.0

		for state, action in trajectory:
			likelihood += np.log()

		return likelihood


	def clone(self):
		"""
		Create a clone of the policy

		return - a clone of the policy
		"""

		# Create a policy with the same state and action space, and copy over
		# the policy map.
		policy = DiscreteStochasticPolicy(self.stateSpace, self.actionSpace)

		policy.policy = self.policy.copy()
		
		return policy
