# See if CuPY is available, otherwise, default to Numpy
try:
	import cupy as np
except:
	import numpy as np

from policy import *



class MaxPolicyGenerator:
	"""
	"""

	def __init__(self, stateSpace, actionSpace):
		"""
		"""

		self.stateSpace = stateSpace
		self.actionSpace = actionSpace


	def generate(self, Q):
		"""
		"""

		policy = DiscreteStochasticPolicy(self.stateSpace, self.actionSpace)
		policy.policy = np.exp(self.beta * Q)
		Z = np.sum(policy.policy, axis=1)
		policy.policy /= np.vstack([Z]*len(self.actionSpace)).T

		return policy


class BoltzmannPolicyGenerator:
	"""
	"""

	def __init__(self, beta, stateSpace, actionSpace):
		"""
		"""

		self.beta = beta

		self.stateSpace = stateSpace
		self.actionSpace = actionSpace


	def generate(self, Q):
		"""
		"""

		policy = DiscreteStochasticPolicy(self.stateSpace, self.actionSpace)
		policy.policy = np.exp(self.beta * Q)
		Z = np.sum(policy.policy, axis=1)
		policy.policy /= np.vstack([Z]*len(self.actionSpace)).T

		return policy


		


class MonteCarloTreeSearchWithTasks:
	"""
	"""

	def __init__(self, mdp, taskLocations):
		"""
		"""

		self.mdp = mdp
		self.tasks = {location: False for location in taskLocations}



class ValueIteration:
	"""
	"""

	def __init__(self, mdp, threshold=0.001, policyGenerator=None):
		"""
		"""

		self.mdp = mdp
		self.threshold = threshold

		self.V = np.zeros((len(self.mdp.stateSpace,)))
		self.Q = np.zeros((len(self.mdp.stateSpace), len(self.mdp.actionSpace)))

		self.policyGenerator = policyGenerator


	def step(self, terminal):
		"""
		"""

		transition = self.mdp.environment.transition.asArray()
		reward = self.mdp.reward.asArray()

		# Calculate the immediate reward for each state/action pair
		Q = np.sum(transition * reward, axis=2)

		# Add the future reward of each state / action pair
		Q += (1.0 - terminal[:,None]) * self.mdp.discount * transition.dot(self.V)

		# Calculate the state value function
		V = np.max(Q, axis=1)

		return V, Q


	def solve(self):
		"""
		"""

		# Create an array indicating which states are terminal
		terminal = [self.mdp.environment.isTerminal(s) for s in self.mdp.stateSpace]
		terminal = np.array(terminal).astype(np.double)

		# Loop until all state values are below threshold
		done = False

		while not done:
			V, Q = self.step(terminal)

			done = np.all(np.abs(self.V - V) < self.threshold)

			self.V = V
			self.Q = Q

		return self.policyGenerator(self)