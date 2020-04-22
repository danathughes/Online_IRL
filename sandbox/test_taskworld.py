from environment import *
from mdp import *
from policy import *
from reward import *
from solvers import *
from transition import *

import matplotlib.pyplot as plt

import time

size = 10
num_tasks = 4

world = Taskworld((size,size),num_tasks)

reward = StaticReward(world.stateSpace, world.actionSpace)
for state in world.stateSpace:
	loc,task = state
	if 

world.addTerminalState((size-1,size-1))
world.addTerminalState((size-1, 0))
world.addTerminalState((4,2))

mdp = MDP(world, reward, 0.95)
solver = ValueIteration(mdp)
solver.solve()

V = np.zeros((size,size))
for state in world.stateSpace:
	x,y = state
	V[x,y] = solver.V[world.stateSpace(state)]

