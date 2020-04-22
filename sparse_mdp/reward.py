import numpy as np
import scipy.sparse
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix, dok_matrix


class AbstractReward:
	 """
	 The AbstractReward defines the common interface for reward functions to
	 implement.  The AbstractReward class makes no assumptions about the type
	 or dimensionality of state and action spaces.  However, derived classes may
	 apply to only one type of space (e.g., discrete state spaces) and
	 dimensionality (e.g., 2D state).
	 """

	 def __init__(self, stateSpace=None, actionSpace=None):
			"""
			Initializes the reward object.  AbstractRewards are not implemented, and
			do not require any sort of initialization.

			stateSpace  - state space of the environment
			actionSpace - action space of the environment
			"""

			self.stateSpace = stateSpace
			self.actionSpace = actionSpace


	 def __call__(self, state, action=None, next_state=None):
			"""
			Calling the object returns the reward.  At a minimum, the returned reward
			is the reward of the passed state, i.e., R(s).  Optionally, the action
			and next_state can be passed, i.e., R(s,a) and R(s,a,s').  Note that the
			call must handle up to three arguments.
			"""

			raise NotImplementedError


	 def __getitem__(self, index):
			"""
			Access rewards through indexing.  This both provides an alternative to
			calling the object to get rewards, as well as allowing slicing, similar
			to numpy arrays
			"""

			raise NotImplementedError


	 def __setitem__(self, index):
			"""
			"""

			raise NotImplementedError


	 def __iter__(self):
			"""
			Initializes iteration of the reward function.  Providing an iterator 
			over ther reward enables iterating over the state-reward pair.
			"""

			raise NotImplementedError


	 def __next__(self):
			"""
			Return the next element when iterating over reward.
			"""

			raise StopIteration










class Tasks:
	"""
	The set of tasks to be performed, and their location in the stateSpace
	"""

	def __init__(self, stateSpace, numTasks):
		"""
		"""

		self.stateSpace = stateSpace
		self.numTasks = numTasks

		self.tasks = {}

		self.observers = []


	def register(self, observer):
		self.observers.append(observer)


	def add(self, state, task):
		"""
		"""

		if state in self.tasks:
			print("WARNING: Replacing existing task")

		if state not in self.stateSpace:
			print("ERROR: %s doesn't exist in state space!" % str(state))

		self.tasks[state] = task
		for observer in self.observers:
			observer.updateTasks(self)


	def remove(self, state):
		"""
		"""

		if state in self.tasks:
			del self.tasks[state]
			for observer in self.observers:
				observer.updateTasks(self)


	def get(self,state):
		"""
		Return the task number for the state, or None
		"""

		return self.tasks[state] if state in self.tasks else None


	def count(self, taskNum):
		return sum([1 if x == taskNum else 0 for x in self.tasks.values()])


	def toList(self):
		"""
		Return a list of tuples of (state, task)
		"""

		return list(self.tasks.items())









class SparseFeatureMap:
	 """
	 """

	 def __init__(self, stateSpace, numFeatures):
			"""
			"""

			self.stateSpace = stateSpace
			self.numFeatures = numFeatures

			self.features = dok_matrix((len(stateSpace), numFeatures), dtype=np.int8)
			self.listeners = []


	 def __informListeners(self):
			"""
			"""

			for listener in self.listeners:
				 listener.notify(self)


	 def __call__(self, state):
			"""
			"""

			return self.features[self.stateSpace(state)]


	 def __getitem__(self, index):
			"""
			"""

			return self.features[index]


	 def __setitem__(self, index, value):
			"""
			"""

			self.features[index] = value

			self.__informListeners()


	 def updateTasks(self, tasks):
	 	"""
	 	"""

	 	for state in self.stateSpace:
	 		if tasks.get(state) is None:
	 			for feature in range(tasks.numTasks):
	 				self.features[self.stateSpace(state), feature] = 0
	 		else:
	 			self.features[self.stateSpace(state), tasks.get(state)] = 1

	 	self.__informListeners()



	 def addListener(self, listener):
			"""
			Inform a listener if a feature has been changed
			"""

			self.listeners.append(listener)


	 def setFeature(self, state, featureNumber):
			"""
			"""

			self.features[self.stateSpace(state), featureNumber] = 1

			self.__informListeners()


	 def clearFeature(self, state, featureNumber):
			"""
			"""

			self.features[self.stateSpace(state), featureNumber] = 0

			self.__informListeners()


	 def asArray(self):
			"""
			"""

			return self.features

















class SparseLinearParametricReward(AbstractReward):
	 """
	 A LinearParametericReward defines rewards for each (state,action,next_state) 
	 triple as a linear combination of state features.  The formulation of the class allows for
	 efficiently representing rewards that are not conditioned on action and / or
	 next state, i.e., R(s) and R(s,a), and optionally broadcasting if desired
	 """

	 def __init__(self, stateSpace, actionSpace, featureMap, parameters, conditionOnNextState=False):
			"""
			featureMap - function that converts state(s) to feature representations
			parameters - array of coefficients for reward function
			"""

			AbstractReward.__init__(self, stateSpace, actionSpace)

			self.featureMap = featureMap
			self.featureMap.addListener(self)

			self.parameters = parameters

			self.reward = [dok_matrix((len(self.stateSpace), len(self.stateSpace))) for a in self.actionSpace]

			self.stale = True


	 def gradient(self):
	 		"""
	 		"""

	 		return dok_matrix(self.featureMap.asArray(), dtype=np.float)


	 def setParameters(self, parameters):
			"""
			"""

			self.parameters = parameters

			self.stale = True


	 def notify(self, featureMap):
			"""
			"""

			self.stale = True


	 def calculateReward(self):
			"""
			"""

			stateFeatures = self.featureMap.asArray()
			stateReward = csr_matrix(stateFeatures.dot(self.parameters))
			stateReward = scipy.sparse.hstack([stateReward.T]*len(self.stateSpace))

			self.reward = [stateReward for _ in self.actionSpace]

			self.stale = False


	 def __call__(self, state, action=None, next_state=None):
			"""
			Calling the object returns the reward.  At a minimum, the returned reward
			is the reward of the passed state, i.e., R(s).  Optionally, the action
			and next_state can be passed, i.e., R(s,a) and R(s,a,s').  Note that the
			call must handle up to three arguments.
			"""

			if self.stale:
				 self.calculateReward()

			return self.reward[self.actionSpace(action)][self.stateSpace(state),self.stateSpace(next_state)]

			
	 def __getitem__(self, index):
			"""
			Access rewards through indexing.  This both provides an alternative to
			calling the object to get rewards, as well as allowing slicing, similar
			to numpy arrays
			"""

			if self.stale:
				 self.calculateReward()

			return self.reward[index[1]][index[0],index[2]]


	 def __iter__(self):
			"""
			Initializes iteration of the reward function.  Providing an iterator 
			over ther reward enables iterating over the state-reward pair.
			"""

			raise NotImplementedError


	 def __next__(self):
			"""
			Return the next element when iterating over reward.
			"""

			raise StopIteration


	 def asArray(self):
			"""
			Return a numpy array representing the reward tensor R(s,a,s')
			"""

			if self.stale:
				 self.calculateReward()

			return self.reward

