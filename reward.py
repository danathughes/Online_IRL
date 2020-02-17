## reward.py
##
## Classes defining reward functions.  Rewards are implemented as callable
## objects, and should be able to handle tensors as well as individual values.



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

      pass


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

   def __init__(self, stateSpace, actionSpace=None):
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



