
class MDP:
	 """
	 An MDP contains the environment, which defines state space, action space,
	 and dynamics / transition functions; reward function; and discount factor.
	 """


	 def __init__(self, environment, reward, discount):
			"""
			Create an MDP with the provided environment, reward function, and
			discount factor.
			"""

			# TODO:  Make sure the environment, reward, and discount are valid
			
			self.environment = environment
			self.reward = reward
			self.discount = discount

			# For convenience, extract the state and action spaces
			self.stateSpace = self.environment.stateSpace
			self.actionSpace = self.environment.actionSpace
