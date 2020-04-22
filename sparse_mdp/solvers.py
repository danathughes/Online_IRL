import numpy as np
import scipy.sparse
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix, dok_matrix

from .policy import *

class SparseValueIteration:
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

			self.observers = []


	 def step(self, terminal, transition=None, reward=None, Q0=None):
			"""
			"""

			if transition is None:
				 transition = [m.tocsr() for m in self.mdp.environment.transition.asArray()]

			if reward is None:
				 reward = [m.tocsr() for m in self.mdp.reward.asArray()]

			# Calculate the immediate reward for each state/action pair, if not provided
#      if Q0 is None:
#         Q0 = np.hstack([t.multiply(r).sum(axis=1) for t,r in zip(transition, reward)])

			# Add the future reward of each state / action pair
			nextQ = self.mdp.discount * np.vstack([t.dot(self.V) for t in transition]).T
			Q = np.array(Q0 + (1.0 - terminal[:,None]) * nextQ)

			# Calculate the state value function
			V = np.max(Q, axis=1)

			return V, Q


	 def register(self, observer):
	 	self.observers.append(observer)


	 def solve(self, update=True):
			"""
			"""

			# Create an array indicating which states are terminal      
			terminal = [self.mdp.environment.isTerminal(s) for s in self.mdp.stateSpace]
			terminal = np.array(terminal).astype(np.double)

			# Get the transition and reward function in csr format
			transition = [m.tocsr() for m in self.mdp.environment.transition.asArray()]
			reward = [m.tocsr() for m in self.mdp.reward.asArray()]

			Q0 = np.hstack([(t.multiply(r)).sum(axis=1) for t,r in zip(transition, reward)])

			# Loop until all state values are below threshold
			done = False

			while not done:
				 V, Q = self.step(terminal, transition, reward, Q0)

				 done = np.all(np.abs(self.V - V) < self.threshold)

				 self.V = V
				 self.Q = Q

			if update:
				for observer in self.observers:
					observer.updateValue(self.V, self.Q)

			return self.policyGenerator(self)












################################################
### Policy Generators
################################################

class BoltzmannPolicyGenerator:
	 """
	 """

	 def __init__(self, beta=1.0):
			"""
			"""

			self.beta = beta


	 def __call__(self, solver):
			"""
			"""

			policy = DiscreteStochasticPolicy(solver.mdp.stateSpace, solver.mdp.actionSpace)

			policy.policy = np.exp(self.beta*solver.Q)
			Z = np.sum(policy.policy, axis=1)
			policy.policy /= np.vstack([Z]*len(solver.mdp.actionSpace)).T

			return policy
