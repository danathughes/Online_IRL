import numpy as np

from sparse_mdp import *
from irl import *
from online_irl import *
from utils import Logger, GridworldVisualizer, GridworldValueVisualizer, AgentVisualizer, MatplotVisualizer, TimeSeriesVisualizer

from world_config_20x20 import *
#from world_config_6x6 import *
import random
import copy
import os
import sys

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



class UAV:
	def __init__(self, mdp, initial_position, fov, tasks):
		self.world_shape = mdp.environment.shape
		self.position = initial_position
		self.fov = [2,2]
		self.steps_since_observed = np.zeros((self.world_shape))
		self.tasks = tasks
		self.detection_rate = 0.01
		self.mdp = mdp


	def observe(self):

		obs_region = []
		for x in range(-self.fov[0], self.fov[0]+1):
			for y in range(-self.fov[1], self.fov[1]+1):
				loc = (self.position[0] + x, self.position[1] + y)
				if loc[0] >= 0 and loc[0] < self.world_shape[0] and loc[1] >= 0 and loc[1] < self.world_shape[1]:
					obs_region.append(loc)

		for loc in obs_region:
			x, y = loc
			self.steps_since_observed[x,y] = 0


	def move(self, action):

		movements = {'up': (0,-1), 'down': (0,1), 'left': (-1,0), 'right': (1,0)}
		x,y = self.position
		x += movements[action][0]
		y += movements[action][1]

		x = min(max(x,0), self.world_shape[0]-1)
		y = min(max(y,0), self.world_shape[1]-1)

		self.position = (x,y)
		self.steps_since_observed += 1


	def selectAction(self):

		movements = {'up': (0,-1), 'down': (0,1), 'left': (-1,0), 'right': (1,0)}
		frontier = {}
		stale = {}

		# Figure out the region covered by the UAV's visual cone
		for action, displacement in movements.items():
			dx,dy = displacement

			frontier_region = []

			for x in range(-self.fov[0], self.fov[0]+1):
				for y in range(-self.fov[1], self.fov[1]+1):
					loc = (self.position[0] + x + dx, self.position[1] + y + dy)
					if loc[0] >= 0 and loc[0] < self.world_shape[0] and loc[1] >= 0 and loc[1] < self.world_shape[1]:
						frontier_region.append(loc)

			frontier[action] = frontier_region

		# How stale is the observation in each direction?
		for action, region in frontier.items():
			stale_count = 0
			for loc in region:
				x,y = loc
				stale_count += self.steps_since_observed[x,y]
			stale[action] = stale_count

		# Softmax to a distribution
		actions = ['up','down','left','right']
		values = [min(stale[a],250) for a in actions]

		probs = np.exp(values) + 1e-8
		probs /= np.sum(probs)

		action = np.random.choice(actions, p=probs)
		return action


	def detect(self, reward_parameters):

		detected_tasks = []
		observation_region = []

		for x in range(-self.fov[0], self.fov[0]+1):
			for y in range(-self.fov[1], self.fov[1]+1):
				loc = (self.position[0] + x, self.position[1] + y)
				if loc[0] >= 0 and loc[0] < self.world_shape[0] and loc[1] >= 0 and loc[1] < self.world_shape[1]:
					observation_region.append(loc)

		for state in observation_region:
			if state in self.mdp.stateSpace and self.tasks.get(state) is None and np.random.random() < self.detection_rate:
				task_num = np.random.randint(0,NUM_TASK_TYPES)
				if len(detected_tasks) + len(self.tasks.toList()) < 40:
					detected_tasks.append((state, task_num))

		return detected_tasks


def run(log_path, visualize):
	#### Create the environment #####
	world = NoisyGridworld((WIDTH,HEIGHT), blocked=blocked, noise = NOISE, TransitionClass=SparseTransition)
	taskWorld = TaskWorld(world)

	tasks = Tasks(world.stateSpace, NUM_TASK_TYPES)

	featureMap = SparseFeatureMap(world.stateSpace, NUM_TASK_TYPES)


	tasks.register(featureMap)
	tasks.register(taskWorld)
	featureMap.updateTasks(tasks)
	taskWorld.updateTasks(tasks)

	# Create the reward
	reward = SparseLinearParametricReward(world.stateSpace, world.actionSpace, featureMap, rewardParameters[0])
	reward.calculateReward()

	# Set up the MDP
	mdp = MDP(world, reward, DISCOUNT)
	solver = SparseValueIteration(mdp, policyGenerator=BoltzmannPolicyGenerator(beta=BETA))
	policy = solver.solve()


	### Set up the IRL ###
	irlReward = SparseLinearParametricReward(world.stateSpace, world.actionSpace, featureMap, np.zeros((NUM_TASK_TYPES,)))
	irlReward.calculateReward()

	irlMdp = MDP(world, irlReward, DISCOUNT)
	irlSolver = SparseValueIteration(irlMdp, policyGenerator=BoltzmannPolicyGenerator(beta=BETA))
	irl = BellmanGradientIRL(irlMdp, irlReward, irlSolver)

	onlineIRL = DecayedBayesianOnlineIRL(irl, DECAY_FACTOR)

	onlineIrlReward = SparseLinearParametricReward(world.stateSpace, world.actionSpace, featureMap, np.zeros((NUM_TASK_TYPES,)))
	onlineIrlReward.calculateReward()

	onlineIrlMdp = MDP(world, onlineIrlReward, DISCOUNT)
	onlineIrlSolver = SparseValueIteration(onlineIrlMdp, policyGenerator=BoltzmannPolicyGenerator(beta=BETA))


	uav=UAV(mdp, (0,1), (2,2), tasks)


	### Create a visualizer for all the components of interest
	if visualize:
		vis=MatplotVisualizer()#(WIDTH,HEIGHT), ['red','blue','green'], blocked=blocked, mdp=mdp, num_steps=1000, r_max=2.1)

		stateValueVisualizer = GridworldValueVisualizer(mdp, title="State Value", v_max=R_MAX)
		onlineIrlValueVisualizer = GridworldValueVisualizer(onlineIrlMdp, title="Online IRL Estimate", v_max=R_MAX)
		staleVisualizer = GridworldValueVisualizer(mdp, title="Staleness", v_max=100)

		mapVisualizer = GridworldVisualizer(mdp, tasks, ['red','blue','green'], blocked=blocked, title="Map")
		playerVisualizer = AgentVisualizer('yellow', (0,0))
		uavVisualizer = AgentVisualizer('cyan', (0,1))
		mapVisualizer.add(playerVisualizer)
		mapVisualizer.add(uavVisualizer)

		rewardParamVisualizer = TimeSeriesVisualizer(['red','blue','green'], max_time=1000, y_max = 2.0, title="Reward Parameters")
		rewardEstimateVisualizer = TimeSeriesVisualizer(['red','blue','green'], max_time=1000, y_max = 2.0, title="Reward Estimate", include_variance=True)
		pseudoestimateVisualizer = TimeSeriesVisualizer(['red','blue','green'], max_time=1000, y_max = 2.0, title="Pseudo Estimate", include_variance=True)
		divergenceVisualizer = TimeSeriesVisualizer(['black'], max_time=1000, y_max = 01000.0, title="KL Divergence")


		vis.add(mapVisualizer, 441)
		vis.add(stateValueVisualizer, 442)
		vis.add(staleVisualizer, 443)
		vis.add(onlineIrlValueVisualizer, 444)

		vis.add(rewardParamVisualizer, 425)
		vis.add(rewardEstimateVisualizer, 426)
		vis.add(pseudoestimateVisualizer, 427)
		vis.add(divergenceVisualizer, 428)


		### Link observers
		solver.register(stateValueVisualizer)
		onlineIrlSolver.register(onlineIrlValueVisualizer)


	log = Logger(log_path,
		         metadata={'size': (WIDTH,HEIGHT),
		                   'noise': NOISE,
	    	               'num_tasks': NUM_TASKS,
	        	           'num_task_types': NUM_TASK_TYPES,
	            	       'beta': BETA,
	                	   'discount': DISCOUNT,
		                   'blocked': blocked})

	# Get the world state
	state = world.current_state
	trajectory = []
	new_intent_time = MIN_NUMBER_STEPS_NEW_INTENT


	for t in range(100):
		# Update the UAV
		uavAction = uav.selectAction()

		uav.move(uavAction)
		uav.observe()

		detected_tasks = uav.detect(onlineIRL.meanReward)
		for task_state, task_num in detected_tasks:
			tasks.add(task_state, task_num)


	for t in range(NUM_STEPS):
		# Update the logger time
		log.setTime(t)

		# Update the reward parameter

		reward.setParameters(rewardParameters[t])
		if visualize:	
			rewardParamVisualizer.add(t, rewardParameters[t])

		log.log('reward_parameters', rewardParameters[t])
		log.log('tasks', tasks.toList())

		# Recalculate the policy
		policy = solver.solve()

		log.log('V', solver.V.copy())
		log.log('Q', solver.Q.copy())

		# Determine the action
		action = policy.selectAction(state)

		log.log('state', state)
		log.log('action', action)

		trajectory.append((state,action))

		log.log('trajectory_likelihood', policy.likelihood(trajectory))

		# Perform the action
		nextState = world.act(action)


		### Update the visualization ###
		if visualize:
			playerVisualizer.updatePosition(state)
			vis.redraw()


		### Check the next state.  If it's on a task, there's work to do
		state = nextState

		updateIRL = False

		# Check if a task has been done 
		if tasks.get(state) is not None:
			onlineIRL.observe(trajectory)
			prev_trajectory = copy.deepcopy(trajectory)
			trajectory = []
			updateIRL = True
			log.log('task_performed', (state, tasks.get(state)))
			tasks.remove(state)


		if len(trajectory) >= MAX_IRL_STEPS:
			onlineIRL.observe(trajectory)
			prev_trajectory = copy.deepcopy(trajectory)
			trajectory = []
			updateIRL = True


		# Update the UAV
		uavAction = uav.selectAction()

		log.log('uav_position', uav.position)
		log.log('uav_action', uavAction)

		uav.move(uavAction)
		uav.observe()

		detected_tasks = uav.detect(onlineIRL.meanReward)
		if len(detected_tasks) > 0:
			if len(trajectory) > 0:
				onlineIRL.observe(trajectory)
				prev_trajectory = copy.deepcopy(trajectory)
				trajectory = []
				updateIRL = True
			for task_state, task_num in detected_tasks:
				tasks.add(task_state, task_num)


		if visualize:
			uavVisualizer.updatePosition(uav.position)
			staleVisualizer.updateGrid(uav.steps_since_observed)		

		if updateIRL:
			log.log('reward_pseudoestimate', onlineIRL.pseudoestimate.copy())
			log.log('reward_pseudovariance', onlineIRL.pseudovariance.copy())
			log.log('reward_estimate_update', onlineIRL.meanReward.copy())
			log.log('reward_variance_update', onlineIRL.varReward.copy())
			log.log('onlineIRL_mu', onlineIRL.mu.copy())
			log.log('onlineIRL_nu', onlineIRL.nu.copy())
			log.log('onlineIRL_alpha', onlineIRL.alpha.copy())
			log.log('onlineIRL_beta', onlineIRL.beta.copy())
			log.log('onlineIRL_KL', onlineIRL.divergence)

			onlineIrlReward.setParameters(onlineIRL.meanReward)
			irlPolicy = irlSolver.solve()
			onlineIrlPolicy = onlineIrlSolver.solve()

			log.log('psuedo_Likelihood', irlPolicy.likelihood(prev_trajectory))
			log.log('pseudo_V', irlSolver.V.copy())
			log.log('pseudo_Q', irlSolver.Q.copy())
			log.log('onlineIRL_likelihood', onlineIrlPolicy.likelihood(prev_trajectory))
			log.log('onlineIRL_V', onlineIrlSolver.V.copy())
			log.log('onlineIrl_Q', onlineIrlSolver.Q.copy())

			if visualize:
				rewardEstimateVisualizer.add(t, onlineIRL.meanReward, variance=onlineIRL.varReward)
				pseudoestimateVisualizer.add(t, onlineIRL.pseudoestimate, variance=onlineIRL.pseudovariance)
				divergenceVisualizer.add(t, [onlineIRL.divergence])

			if onlineIRL.divergence >= NEW_INTENT_THRESHOLD and t >= new_intent_time:
				new_intent_time = t + MIN_NUMBER_STEPS_NEW_INTENT
				log.log('final_intent_reward_parameters', onlineIRL.meanReward.copy())
				onlineIRL.init_hyperparameters()
				log.log('new_intent', True)



	# Save the log
	log.save()

#	if visualize:
#		vis.close()


if __name__ == '__main__':
	
	path = sys.argv[1]
	start_num = int(sys.argv[2])
	end_num = int(sys.argv[3])
	if len(sys.argv) > 4:
		visualize = eval(sys.argv[4])
	else:
		visualize = False

	if not os.path.exists(path):
		os.makedirs(path)
	path_template = path + "run_%d.pkl"
	for run_num in range(start_num, end_num):
		print("Experiment %d" % run_num)
		run(path_template % run_num, visualize) 
