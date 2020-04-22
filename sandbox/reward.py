## reward.py
##
## Classes defining reward functions.  Rewards are implemented as callable
## objects, and should be able to handle tensors as well as individual values.

# See if CuPY is available, otherwise, default to Numpy
try:
   import cupy as np
except:
   import numpy as np

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



class StaticReward(AbstractReward):
   """
   A StaticReward defines a fixed reward for each (state,action,next_state) 
   triple, i.e., a fixed R(s,a,s').  The formulation of the class allows for
   efficiently representing rewards that are not conditioned on action and / or
   next state, i.e., R(s) and R(s,a), and optionally broadcasting if desired
   """



   def __init__(self, stateSpace, actionSpace=None, conditionOnNextState=False):
      """
      
      """

      AbstractReward.__init__(self, stateSpace, actionSpace)

      self.reward = np.zeros((len(self.stateSpace), len(self.actionSpace), len(self.stateSpace)))


   def __call__(self, state, action=None, next_state=None):
      """
      Calling the object returns the reward.  At a minimum, the returned reward
      is the reward of the passed state, i.e., R(s).  Optionally, the action
      and next_state can be passed, i.e., R(s,a) and R(s,a,s').  Note that the
      call must handle up to three arguments.
      """

      return self.reward[self.stateSpace(state), self.actionSpace(action), self.stateSpace(next_state)]

      
   def setReward(self, *args):
      """
      Set the reward for the provided state / state-action pair / 
      state-action-next_state triple.

      """

      state, action, next_state, reward = args

      self.reward[self.stateSpace(state), self.actionSpace(action), self.stateSpace(next_state)] = reward


   def __getitem__(self, index):
      """
      Access rewards through indexing.  This both provides an alternative to
      calling the object to get rewards, as well as allowing slicing, similar
      to numpy arrays
      """

      return self.reward[index]


   def __setitem__(self, index, value):
      """
      """

      self.reward[index] = value


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

      return self.reward



class FeatureMap:
   """
   """

   def __init__(self, stateSpace, numFeatures):
      """
      """

      self.stateSpace = stateSpace
      self.numFeatures = numFeatures

      self.features = np.zeros((len(stateSpace), numFeatures), dtype=np.int8)
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


   def addListener(self, listener):
      """
      Inform a listener if a feature has been changed
      """

      self.listeners.append(listener)


   def setFeature(self, state, featureNumber):
      """
      """

      self.features[self.stateSpace(state), featureNumber] = 1.0

      self.__informListeners()

   def clearFeature(self, state, featureNumber):
      """
      """

      self.features[self.stateSpace(state), featureNumber] = 0.0

      self.__informListeners()


   def asArray(self):
      """
      """

      return self.features


class LinearParametricReward(AbstractReward):
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

      self.reward = np.zeros((len(self.stateSpace), len(self.actionSpace), len(self.stateSpace)))

      self.stale = True


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
      stateReward = np.sum(stateFeatures*self.parameters, axis=1)

      stateReward = np.reshape(stateReward,(len(self.stateSpace),1,1))
      self.reward = np.tile(stateReward, (1, len(self.actionSpace), len(self.stateSpace)))

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

      return self.reward[self.stateSpace(state), self.actionSpace(action), self.stateSpace(next_state)]

      
   def __getitem__(self, index):
      """
      Access rewards through indexing.  This both provides an alternative to
      calling the object to get rewards, as well as allowing slicing, similar
      to numpy arrays
      """

      if self.stale:
         self.calculateReward()

      return self.reward[index]


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