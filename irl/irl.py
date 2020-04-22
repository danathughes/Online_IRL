import numpy as np

from .bellman_max_approximation import *
from .optimizers import *

class BayesianIRL:
	"""
	"""

	def __init__(self, mdp, reward, solver, step_size = 0.05, R_max = 2.0):
		"""
		"""

		self.mdp = mdp
		self.solver = solver
		self.reward = reward

		self.step_size = step_size
		self.R_max = R_max

		self.reward_params_with_likelihoods = []


	def step(self, trajectory):
		"""
		"""

		# What is the previous likelihood of the trajectory				
		policy = self.solver.solve(update=False)
		old_likelihood = policy.likelihood(trajectory)
		old_parameters = self.reward.parameters.copy()

		# Take a random step along one of the reward parameter directions
		# Randomly add or subtract by the step size along the dimension
		idx = np.random.randint(self.reward.parameters.shape[0])
		sign = 1.0 if np.random.random() < 0.5 else -1.0
		self.reward.parameters[idx] += sign * self.step_size
		self.reward.parameters[idx] = min(max(self.reward.parameters[idx],-self.R_max),self.R_max)

		# Update the reward parameters and recompute the value functions and
		# policy and updated likelihood
		self.reward.setParameters(self.reward.parameters)
		policy = self.solver.solve(update=False)
		new_likelihood = policy.likelihood(trajectory)

		# What is the probability ratio?
		prob_ratio = min(1.0, np.exp(new_likelihood - old_likelihood))

		# Check if the new reward parameters should be accepted
		if np.random.random() > prob_ratio:         # Failed the test
			self.reward.setParameters(old_parameters)


	def update_reward(self, trajectory, num_iter=100):
		"""
		"""

		# Copy the original parameters, V and Q values
		originalParameters = self.reward.parameters.copy()
		originalV = self.solver.V
		originalQ = self.solver.Q

		# Reset the list of reward params with likelihoods
		self.reward_params_with_likelihoods = []

		# Get the likelihood of the trajectory under the initial reward
		policy = self.solver.solve(update=False)
		self.reward_params_with_likelihoods.append((self.reward.parameters.copy(), policy.likelihood(trajectory)))
		best_reward = self.reward_params_with_likelihoods[0]

		for i in range(num_iter):
			self.step(trajectory)
			policy = self.solver.solve(update=False)
			self.reward_params_with_likelihoods.append((self.reward.parameters.copy(), policy.likelihood(trajectory)))
			if self.reward_params_with_likelihoods[-1][1] > best_reward[1]:
				best_reward = self.reward_params_with_likelihoods[-1]

		self.reward.setParameters(best_reward[0])
		policy = self.solver.solve(update=False)























class BellmanGradientIRL:
	"""
	"""

	def __init__(self, mdp, reward, solver,
				 optimizer=GradientAscentOptimizer(0.25, difference_threshold=0.1, min_iterations=5, max_iterations=50),
		         gradient_difference_threshold = 0.1,
		         max_gradient_iterations = 50,
		         min_gradient_iterations = 5,
		         approxMax = PNorm(20),
		         R_max = 2.0):
		"""
		"""

		self.mdp = mdp
		self.solver = solver
		self.reward = reward

#		self.learning_rate = learning_rate
		self.R_max = R_max

#		self.parameter_difference_threshold = parameter_difference_threshold
#		self.max_parameter_iterations = max_parameter_iterations
#		self.min_parameter_iterations = min_parameter_iterations

		self.optimizer = optimizer

		self.gradient_difference_threshold = gradient_difference_threshold
		self.max_gradient_iterations = max_gradient_iterations
		self.min_gradient_iterations = min_gradient_iterations

		self.reward_params_with_likelihoods = []

		self.approxMax = approxMax


	def bellmanGradientIteration(self, 
		                         terminalStates = None,
		                         rewardGradient = None,
		                         transition = None):
		"""
		Calculate the gradient of the bellman function
		"""

		# Where to maintain the gradient of V and Q
		VGrad = np.zeros((len(self.mdp.stateSpace),) + self.reward.parameters.shape)
		QGrad = np.zeros((len(self.mdp.stateSpace), len(self.mdp.actionSpace)) + self.reward.parameters.shape)

		# Conditions for finishing
		done = False
		iter_num = 0

		# Calculate terminal state tensor, if necessary
		if terminalStates is None:
			terminalStates = np.array([self.mdp.environment.isTerminal(s) for s in self.mdp.stateSpace]).astype(np.float)

		# Get the gradient of the reward, if necessary
		if rewardGradient is None:
			rewardGradient = np.array(self.reward.gradient().todense())

		# Get the transition as an array
		if transition is None:
			transition = self.mdp.environment.transition.asArray()

		# Calculate the gradient of the approxMax function
		approxMaxGrad = self.approxMax.gradient(self.solver.Q)

		while not done:
			# Calculate Tgradient
			# Next State Return
			nextStateReturn = rewardGradient + (1.0 - terminalStates[:,None]) * self.mdp.discount * VGrad

			# Dot next state return with each P^a_{s,s'} to get T
			TGrad = np.array([trans*nextStateReturn for trans in transition]) 
			# T is in terms of T(a,s,theta), swap axes to get T(s,a,theta)
			TGrad = np.swapaxes(TGrad,0,1)

			# Multiply by the approxMax gradient
			TGrad *= approxMaxGrad[:,:,None]

			# Calculate Vg'
			VGradPrime = np.sum(TGrad, axis=1)

			# Check if should finish
			diff = np.abs(VGrad - VGradPrime)
			done = np.all(diff < self.gradient_difference_threshold)
			done = done or iter_num > self.max_gradient_iterations
			done = done and iter_num > self.min_gradient_iterations

			VGrad = VGradPrime
			iter_num += 1


		# Calculate VGrad, now calculate QGrad
		# Dot next state return with each P^a_{s,s'} to get T
		QGrad = np.array([trans*nextStateReturn for trans in transition]) 
		QGrad = np.swapaxes(QGrad,0,1)

		return VGrad, QGrad							



	def likelihood_gradient(self, trajectory):
		"""
		"""

		# Calculate the policy
		policy = self.solver.solve(update=False)

		beta = self.solver.policyGenerator.beta

		# Calculate the gradients of the V and Q functions w.r.t theta
		VGrad, QGrad = self.bellmanGradientIteration()

		likelihood_gradients = []

		# Calculate the likelihood gradient for each state/action pair in the
		# trajectory
		for s,a in trajectory:
			likelihoodGrad = beta*QGrad[self.mdp.stateSpace(s),self.mdp.actionSpace(a)]

			likelihoodGrad -= np.sum(beta*policy.getActionDistribution(s)[:,None]*QGrad[self.mdp.stateSpace(s),:],axis=0)

#			for ap in self.mdp.actionSpace:
#				likelihoodGrad -= self.beta*policy.getActionProbability(s,ap)*QGrad[mpd.stateSpace(s),mdp.actionSpace(ap)]

			likelihood_gradients.append(likelihoodGrad)

		likelihood_gradients = np.array(likelihood_gradients)

		return np.sum(likelihood_gradients, axis=0)



	def update_reward(self, trajectory, prior=None, num_iter=100):
		"""
		"""

		# Copy the original parameters, V and Q values
		originalParameters = self.reward.parameters.copy()
		originalV = self.solver.V
		originalQ = self.solver.Q

		# Conditions to see if the algorithm update is done
		iter_num = 0
		done = False

		self.params_with_likelihoods = []

		self.optimizer.initialize(self.reward.parameters)

		while not self.optimizer.done():

#			old_parameters = self.reward.parameters.copy()

			# Calculate the gradient and apply to the opimizer
			gradient = self.likelihood_gradient(trajectory)

			# Was there a prior?
			if prior is not None:
				mu, sigma = prior
				gradient -= (self.reward.parameters - mu)/sigma

			self.optimizer.update(gradient)

			# Update the reward parameters -- bound by R_max
#			new_parameters = self.reward.parameters + self.learning_rate * gradient
#			new_parameters[new_parameters > self.R_max] = self.R_max
#			new_parameters[new_parameters < -self.R_max] = -self.R_max

#			self.reward.setParameters(new_parameters)

			# Indicate that the reward parameters have been modified
			self.reward.stale = True

			# Store the parameters and likelihoods
			policy = self.solver.solve(update=False)
			self.params_with_likelihoods.append((self.reward.parameters.copy(), policy.likelihood(trajectory)))

			# Are we done?
#			diff = np.abs(new_parameters - old_parameters)
#			done = np.all(diff < self.parameter_difference_threshold)
#			done = done or iter_num > self.max_parameter_iterations
#			done = done and iter_num > self.min_parameter_iterations

#			iter_num += 1

