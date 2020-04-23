import numpy as np



class DecayedBayesianOnlineIRL:
	"""
	"""

	def __init__(self, irl, decayRate,
		         alpha0 = 2.0, beta0 = 4.0,
		         mu0 = 0.0, nu0 = 5):
		"""
		irl - the base IRL estimator
		decayRate - decay rate of the parameters
		alpha0 - initial value of alpha
		beta0 - initial value of beta
		mu0 - initial value of mu
		nu0 - initial value of nu
		"""

		self.baseIRL = irl
		self.decayRate = decayRate
		self.reward_shape = irl.reward.parameters.shape

		self.initial_parameters = (alpha0, beta0, mu0, nu0)

		self.divergence = 0.0

		self.init_hyperparameters()


	def init_hyperparameters(self):
		"""
		Initialize 
		"""

		self.alpha = np.zeros(self.reward_shape) + self.initial_parameters[0]
		self.beta = np.zeros(self.reward_shape) + self.initial_parameters[1]
		self.mu = np.zeros(self.reward_shape) + self.initial_parameters[2]
		self.nu = np.zeros(self.reward_shape) + self.initial_parameters[3]

		# Reward mean and variance accessible
		self.meanReward = self.mu
		self.varReward = self.beta / (self.alpha - 1.0)

		self.pseudoestimate = np.zeros(self.reward_shape)
		self.pseudovariance = np.zeros(self.reward_shape)


	def decayHyperparameters(self, numSteps):
		"""
		Decay the hyperparameters by the number of steps
		"""

		# Calculate the total decay over the number of steps
		decay = self.decayRate**numSteps

		self.alpha = decay * (self.alpha - 1.0) + 1.0
		self.beta *= decay
		self.nu *= decay

		# Recalculate the variance
		self.varReward = self.beta / (self.alpha - 1.0)



	def estimateRewardVariance(self, trajectory, prior=None, num_samples = 200):
		"""
		Estimate the reward variance
		"""

		# Store the previous reward parameters
		policy = self.baseIRL.solver.solve(update=False)
		reward_likelihood = policy.likelihood(trajectory)
		old_parameters = self.baseIRL.reward.parameters.copy()

		# Sample a set of reward parameters from a universal distribution
		R_max = self.baseIRL.R_max
		rewardParameterSamples = np.random.uniform(-R_max, R_max, ((num_samples,) + self.reward_shape))

		# Calculate the likelihood of each
		sample_likelihoods = np.zeros((num_samples,))

		for i in range(num_samples):
			self.baseIRL.reward.setParameters(rewardParameterSamples[i])
			policy = self.baseIRL.solver.solve(update=False)
			sample_likelihoods[i] = policy.likelihood(trajectory)

# Include a prior?
#			if prior is not None:
#				dist = np.sum((self.mdp.env.reward.parameters - mu)**2/sigma)
#				prior_likelihood = -0.5*dist
#			else:
#				prior_likelihood = 0.0


		# Restore the old parameters
		self.baseIRL.reward.setParameters(old_parameters)


		# Calculate the weights of each reward sample, and the weighted
		# variance
		weights = np.exp(sample_likelihoods - reward_likelihood)
		weights = weights / np.sum(weights)
		weightedVariances = ((weights**2)[:,None])*(rewardParameterSamples - old_parameters)**2
		weightedVariances = np.sum(weightedVariances,axis=0) #/ np.sum(weights)

		return weightedVariances


	def kl(self, mu1, sig1, mu2, sig2):
		"""
		Calculate the KL divergence of two Gaussians

		mu1 - mean of gaussian distribution #1
		sig1 - standard deviations
		mu2 - mean of gaussian distribution #2
		sig2 - standard deviation
		"""

		sig1 = np.diag(sig1 + 1e-8)
		sig2 = np.diag(sig2 + 1e-8)

		det1 = np.linalg.det(sig1)
		det2 = np.linalg.det(sig2)

		sig2_inv = np.linalg.inv(sig2)

		mu_diff = (mu2 - mu1)

		A = np.log(det2 / det1)
		B = np.trace(sig2_inv.dot(sig1))
		C = mu_diff.dot(sig2_inv.dot(mu_diff))
		N = self.reward_shape[0]

		return 0.5*(A+B+C-N)


	def observe(self, trajectory):
		"""
		"""

		# Save the current (old) reward parameter estimates to calculate KL divergence
		oldMeanReward = self.meanReward.copy()
		oldVarReward = self.varReward.copy()

		# Set the reward parameters to the current estimate
		self.baseIRL.reward.setParameters(self.meanReward.copy())

		# Update the reward function with the current trajectory
		self.baseIRL.update_reward(trajectory, prior=(oldMeanReward, np.sqrt(oldVarReward)))

		self.pseudoestimate = self.baseIRL.reward.parameters.copy()

		self.stateValueEstimate = self.baseIRL.solver.V.copy()

		# Estimate the variance
		self.pseudovariance = self.estimateRewardVariance(trajectory)

		# Update the hyperparameters
		# Decay current parameters to reduce prior evidence weight
		N = len(trajectory)
		self.decayHyperparameters(N)

		# Update hyperparameters with new evidence
		self.alpha += 0.5*N
		self.beta += 0.5*N*self.pseudovariance
		self.beta += 0.5*((N*self.nu)/(N+self.nu))*(self.pseudoestimate - self.mu)**2
		self.mu = (self.nu * self.mu + N * self.pseudoestimate)/(N + self.nu)
		self.nu += N

		# Calculate the new reward mean and variance
		self.meanReward = self.mu
		self.varReward = self.beta / (self.alpha - 1.0)

		self.divergence = self.kl(oldMeanReward, oldVarReward, self.pseudoestimate, self.pseudovariance)
