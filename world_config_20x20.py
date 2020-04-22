import numpy as np

# Properties of the gridworld
blocked=np.array([[0,0,0,0,0,0,1,0,0,0,1,1,0,0,1,0,0,0,1,0],
				  [0,0,0,0,1,1,1,0,0,0,1,0,0,0,1,0,0,1,1,0],
				  [1,1,0,0,0,0,1,0,0,0,1,1,1,0,1,0,0,1,0,0],
				  [1,0,0,0,0,0,1,0,0,0,0,1,0,0,1,1,0,1,0,1],
				  [1,0,0,0,1,1,1,0,0,0,0,0,0,1,1,0,0,1,0,1],
				  [1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0],		 		 		 		 
				  [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0],
				  [0,0,1,1,1,1,1,1,0,0,1,0,0,1,0,0,1,0,0,0],
				  [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				  [0,0,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,1,1,1],
				  [0,0,0,0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0],
				  [0,1,1,0,0,1,0,0,0,0,1,0,0,0,0,1,1,1,0,0],
				  [0,1,0,0,0,1,1,1,1,1,1,0,0,0,0,1,0,0,0,0],
				  [1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
				  [0,0,0,1,1,0,0,0,0,0,0,0,1,0,0,1,0,1,0,0],
				  [0,0,0,0,1,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0],
				  [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0],
				  [0,0,1,1,1,1,1,1,1,0,0,1,0,0,1,0,0,0,0,0],
				  [0,0,0,0,1,0,0,0,0,0,0,1,0,0,1,1,1,0,0,0],
				  [0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]],dtype=np.int)


WIDTH = blocked.shape[0]
HEIGHT = blocked.shape[1]

# Properties of the tasks to be performed
NUM_TASK_TYPES = 3
NUM_TASKS = 20


STEP_COUNTS = [400,200,400]
NUM_STEPS = sum(STEP_COUNTS)

# MDP Parameters
NOISE = 0.2
BETA = 20.0
DISCOUNT = 0.95

# Create the parameters
rewardParameters = np.zeros((NUM_STEPS, NUM_TASK_TYPES))
R_MAX = 2.0

for t in range(STEP_COUNTS[0]):
	rewardParameters[t,0] = 1.0 + np.sin(np.pi*t / 200.)
	rewardParameters[t,1] = 1.0 - np.cos(np.pi*t / 400.)
	rewardParameters[t,2] = -0.25

for t in range(STEP_COUNTS[1]):
	rewardParameters[t + sum(STEP_COUNTS[:1]),0] = -1.0
	rewardParameters[t + sum(STEP_COUNTS[:1]),1] = -1.0
	rewardParameters[t + sum(STEP_COUNTS[:1]),2] = 2.0

for t in range(STEP_COUNTS[2]):
	rewardParameters[t + sum(STEP_COUNTS[:2]),0] = np.cos(np.pi * t / 200.)
	rewardParameters[t + sum(STEP_COUNTS[:2]),1] = np.cos(np.pi*t / 400.)
	rewardParameters[t + sum(STEP_COUNTS[:2]),2] = -2.0

# IRL Parameters
MAX_IRL_STEPS = 25
DECAY_FACTOR=0.985
