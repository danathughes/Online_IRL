import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from sparse_mdp import *
from world_config_20x20 import *



def createTasks(stateSpace, num_task_types):
	"""
	"""

	tasks = Tasks(stateSpace, num_task_types)

	for i in range(NUM_TASKS):
		task_num = np.random.randint(0,NUM_TASK_TYPES)
		state = random.choice([s for s in stateSpace])
		while tasks.get(state) is not None:
			state = random.choice([s for s in stateSpace])
		tasks.add(state, task_num)

	return tasks


class TaskWorld:
	"""
	"""

	def __init__(self, world):
		"""
		"""

		self.world = world

	def updateTasks(self, tasks):
		# Somewhat of a hack---clear all the terminal 
		# states in the world, and set the terminal states
		# to those in the tasks

		self.world.terminalStates.clear()
		for task in tasks.tasks.keys():
			self.world.addTerminalState(task)


def getV(task_list, rewardParameters, policy):
	#### Create the environment #####
	world = NoisyGridworld((WIDTH,HEIGHT), blocked=blocked, noise = NOISE, TransitionClass=SparseTransition)
	taskWorld = TaskWorld(world)

	tasks = Tasks(world.stateSpace,3)
	for s, t in task_list:
		tasks.add(s,t)

	featureMap = SparseFeatureMap(world.stateSpace, NUM_TASK_TYPES)

	tasks.register(featureMap)
	tasks.register(taskWorld)
	featureMap.updateTasks(tasks)
	taskWorld.updateTasks(tasks)

	# Create the reward
	reward = SparseLinearParametricReward(world.stateSpace, world.actionSpace, featureMap, rewardParameters)
	reward.calculateReward()

	# Set up the MDP
	mdp = MDP(world, reward, DISCOUNT)
	solver = ILESolver(mdp, policyGenerator=BoltzmannPolicyGenerator(beta=BETA))
	_ = solver.solve(policy)

	return solver.V


def unpackFile(pickle_filename):

	with open(pickle_filename, 'rb') as pickle_file:
		data = pickle.load(pickle_file)

	return data


def loadFiles(path):

	filenames = os.listdir(path)

	dataset = []

	for name in filenames:
		file_path = os.path.join(path, name)
		dataset.append(unpackFile(file_path))

	return dataset


def get_reward_parameters(dataset, reward_key='reward_parameters'):

	all_rewards = []
	all_times = []

	for run in dataset:
		rewards = []
		times = []

		for t in run['data']:
			if reward_key in run['data'][t].keys():
				rewards.append(run['data'][t][reward_key])
				times.append(t)

		all_rewards.append(np.array(rewards))
		all_times.append(np.array(times))

	return all_times, all_rewards


def get_extended_Qs(time, Q, max_time=1000):

	cur_Q = np.zeros(Q[0].shape)

	Qs = []

	idx = 0

	for t in range(max_time):
		# Have we gotten to the next reward?
		if idx < len(time) and t == time[idx]:

			cur_Q = Q[idx,:]
			idx += 1

		Qs.append(cur_Q.copy())

	return Qs


def getILE(dataset):

	# Extract the needed info
	_,reward_parameters = get_reward_parameters(dataset)
	_,task_lists = get_reward_parameters(dataset,'tasks')
	_,V = get_reward_parameters(dataset,'V')
	update_times, irlQ = get_reward_parameters(dataset,'onlineIrl_Q')

	all_ile = []

	for i in range(len(irlQ)):
		print(i)
		extendedIRL_Q = get_extended_Qs(update_times[i], irlQ[i])
		
		ile = []
		for t in range(len(reward_parameters[0])):
			policy = np.exp(20*extendedIRL_Q[t])
			policy /= np.sum(policy, axis=1)[:,None]

			V_pi = getV(task_lists[i][t], reward_parameters[i][t], policy)

			dV = V[i][t] - V_pi

			_ile = (1./V_pi.shape[0])*np.sqrt(np.sum(dV**2))

			ile.append( (_ile, np.mean(V[i][t]), np.max(V[i][t]) ) )
		print(len(ile))

		all_ile.append(ile)

	return all_ile





# Human 2 - 1.5%
# Human 3 - 51.5% / 3.46% / 0.5
# Human 4 - 6.67%
# Human 5 - 27.27% / 1.68%
# Human 6 - 39.47% / 1.64%
# Human 7 - 72.0% / 6.4% 0.1
# Human 8 - 85.29% / 7.09% / 0.2
#dataset = loadFiles('experiments/human_monitoring/')
dataset = loadFiles('experiments/human8/')
