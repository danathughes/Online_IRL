from environment import *
from mdp import *
from policy import *
from reward import *
from solvers import *
from transition import *

import matplotlib.pyplot as plt

import time

size = 10

blocked=np.array([[0,0,0,0,0,0,1,0,0,0],
				  [0,0,0,1,1,1,1,0,0,0],
				  [0,0,0,0,0,0,1,0,0,0],
				  [1,0,0,0,0,0,1,0,0,0],
				  [1,0,0,0,1,1,1,0,0,0],
				  [1,0,0,0,0,0,0,0,0,0],		 		 		 		 
				  [0,0,0,0,0,0,0,0,0,0],
				  [0,0,1,1,1,1,1,1,1,1],
				  [0,0,1,0,0,0,0,0,0,0],
				  [0,0,0,0,0,0,0,0,0,0]],dtype=np.int)

world = NoisyGridworld((size,size),blocked=blocked,noise=0.25)

featureMap = FeatureMap(world.stateSpace, 3)
featureMap.setFeature((4,2), 0)
featureMap.setFeature((size-1, size-1), 1)
featureMap.setFeature((size-1, 0), 2)

reward = LinearParametricReward(world.stateSpace, world.actionSpace, featureMap, np.array((0.6,1.0,-1.0)))
reward.calculateReward()
reward.reward -= 0.05

world.addTerminalState((size-1,size-1))
world.addTerminalState((size-1, 0))
world.addTerminalState((4,2))

mdp = MDP(world, reward, 0.95)
solver = ValueIteration(mdp)
solver.solve()


V = np.zeros((size,size)) - 3
for state in world.stateSpace:
	x,y = state
	V[x,y] = solver.V[world.stateSpace(state)]

plt.imshow(V)
plt.show()