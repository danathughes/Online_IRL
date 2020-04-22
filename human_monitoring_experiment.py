import numpy as np

from sparse_mdp import *
from irl import *
from online_irl import *
from utils import Logger, GridworldVisualizer, GridworldValueVisualizer, AgentVisualizer, MatplotVisualizer

from world_config_20x20 import *
import random
import copy

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


def run(log_path, visualize):
	#### Create the environment #####
	world = NoisyGridworld((WIDTH,HEIGHT), blocked=blocked, noise = NOISE, TransitionClass=SparseTransition)
	taskWorld = TaskWorld(world)

	tasks = createTasks(world.stateSpace, NUM_TASK_TYPES)

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


	### Create a visualizer for all the components of interest
	if visualize:
		vis=MatplotVisualizer()#(WIDTH,HEIGHT), ['red','blue','green'], blocked=blocked, mdp=mdp, num_steps=1000, r_max=2.1)

		stateValueVisualizer = GridworldValueVisualizer(mdp, title="State Value", v_max=R_MAX)
		irlValueVisualizer = GridworldValueVisualizer(irlMdp, title="Pseudoestimate", v_max=R_MAX)
		onlineIrlValueVisualizer = GridworldValueVisualizer(onlineIrlMdp, title="Online IRL Estimate", v_max=R_MAX)

		mapVisualizer = GridworldVisualizer(mdp, tasks, ['red','blue','green'], blocked=blocked, title="Map")
		playerVisualizer = AgentVisualizer('yellow', (0,0))
		mapVisualizer.add(playerVisualizer)

		vis.add(mapVisualizer, 141)
		vis.add(stateValueVisualizer, 142)
		vis.add(irlValueVisualizer, 143)
		vis.add(onlineIrlValueVisualizer, 144)

		### Link observers
		solver.register(stateValueVisualizer)
		irlSolver.register(irlValueVisualizer)
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



	for t in range(NUM_STEPS):
		# Update the logger time
		log.setTime(t)

		# Update the reward parameter

		reward.setParameters(rewardParameters[t])

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

			tasks.remove(state)

			# Add a random task somewhere there isn't one
			task_state = state
			while task_state == state or tasks.get(task_state) is not None:
				task_state = random.choice([s for s in world.stateSpace])
			task_num = np.random.randint(0, NUM_TASK_TYPES)
			for x in range(NUM_TASK_TYPES):
				if tasks.count(x) == 0:
					task_num = x

			tasks.add(task_state, task_num)

			featureMap.setFeature(task_state, task_num)

		if len(trajectory) >= MAX_IRL_STEPS:
			onlineIRL.observe(trajectory)
			trajectory = []
			prev_trajectory = copy.deepcopy(trajectory)
			updateIRL = True

		if updateIRL:
			log.log('reward_pseudoestimate', onlineIRL.pseudoestimate.copy())
			log.log('reward_pseudovariance', onlineIRL.pseudoestimate.copy())
			log.log('reward_estimate_update', onlineIRL.meanReward.copy())
			log.log('reward_variance_update', onlineIRL.varReward.copy())
			log.log('onlineIRL_mu', onlineIRL.mu.copy())
			log.log('onlineIRL_nu', onlineIRL.nu.copy())
			log.log('onlineIRL_alpha', onlineIRL.alpha.copy())
			log.log('onlineIRL_beta', onlineIRL.beta.copy())
			log.log('onlineIRL_KL', onlineIRL.divergence)

			print("Time Step %d:" % t)
			print("  Reward Parameters:", rewardParameters[t])
			print()
			print("  IRL Update:")
			print("    Reward Pseudoestimate: ", onlineIRL.pseudoestimate)
			print("    Reward Pseudovariance: ", onlineIRL.pseudovariance)
			print("  OnlineIRL Update:")
			print("    Updated Reward Estimate: ", onlineIRL.meanReward)
			print("    Updated Reward Variance: ", onlineIRL.varReward)
			print("    KL Divergence: ", onlineIRL.divergence)
			print()
			print()


			onlineIrlReward.setParameters(onlineIRL.meanReward)
			irlPolicy = irlSolver.solve()
			onlineIrlPolicy = onlineIrlSolver.solve()

			log.log('psuedo_Likelihood', irlPolicy.likelihood(prev_trajectory))
			log.log('pseudo_V', irlSolver.V.copy())
			log.log('pseudo_Q', irlSolver.Q.copy())
			log.log('onlineIRL_likelihood', onlineIrlPolicy.likelihood(prev_trajectory))
			log.log('onlineIRL_V', onlineIrlSolver.V.copy())
			log.log('onlineIrl_Q', onlineIrlSolver.Q.copy())



	# Save the log
	log.save()

	if visualize:
		vis.close()


if __name__ == '__main__':
	for i in range(1):
		print("Experiment %d" % i)
		run('dummy.pkl', False)
#		run('human_monitoring_experiment_20x20_20tasks_no_intent_recognition_%d.pkl'%i)