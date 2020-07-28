import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

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


def get_extended_rewards(time, reward, max_time=1000):

	cur_reward = np.array((0,0,0))

	rewards = []

	idx = 0

	for t in range(max_time):
		# Have we gotten to the next reward?
		if idx < len(time) and t == time[idx]:

			cur_reward = reward[idx,:]
			idx += 1

		rewards.append(cur_reward.copy())

	return rewards



def kl(mu1, sig1, mu2, sig2):
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
	N = mu1.shape[0]

	return 0.5*(A+B+C-N)


def true_intent_change(times, lower, upper):
	count = 0
	for t in times:
		if t >= lower and t <= upper:
			count += 1

	return count


def make_kl_plot(dataset):

	irl_update_times, kl = get_reward_parameters(dataset, 'onlineIRL_KL')
#	extended_kl = np.array([np.array(get_extended_rewards(irl_update_times[i], kl[i])) for i in range(100)])

	plt.plot(kl[0])
	plt.show()

	intent_change_times, _ = get_reward_parameters(dataset, 'new_intent')

	all_update_times = [t for times in irl_update_times for t in times]
	all_intent_change_times = [t for times in intent_change_times for t in times]

	print("Number Updates: ", len(all_update_times))
	print("Number Intent Changes: ", len(all_intent_change_times))

	true_changes = []


	for i in range(51):
		counts = true_intent_change(all_intent_change_times, 400, 400+i) + true_intent_change(all_intent_change_times, 600, 600+i)
		true_changes.append(counts)

	false_changes = [len(all_intent_change_times) - c for c in true_changes]

	true_positive_rate = [100*float(c) / (2*len(kl)) for c in true_changes]
	false_positive_rate = [100*float(f) / len(all_update_times) for f in false_changes]

	plt.plot(range(51), true_positive_rate, '-r', linewidth=2, label='True Positives')
	plt.plot(range(51), false_positive_rate, '-b', linewidth=2, label='False Positives')
	plt.legend(loc='upper left')
	plt.xlabel('Window Size (Time Steps)', fontsize=14)
	plt.ylabel('Rate (%)', fontsize=14)
	plt.xlim(0,50)
	plt.ylim(0,100)
	plt.show()

	print("True positive rates:")
	print(true_positive_rate)

	print("False positive rates:")
	print(false_positive_rate)



def make_reward_plot(dataset):

	human_reward_time , human_rewards = get_reward_parameters(dataset)
	estimated_reward_times, estimated_rewards = get_reward_parameters(dataset, 'reward_estimate_update')
	estimated_reward_times_std, estimated_rewards_std = get_reward_parameters(dataset, 'reward_variance_update')


	extended_reward_estimates = np.array([np.array(get_extended_rewards(estimated_reward_times[i], estimated_rewards[i])) for i in range(len(estimated_reward_times))])
	extended_reward_variances = np.array([np.array(get_extended_rewards(estimated_reward_times[i], estimated_rewards_std[i])) for i in range(len(estimated_reward_times))])



	mean_reward_estimates = np.mean(extended_reward_estimates, axis=0)
	mean_reward_std = np.mean(extended_reward_variances, axis=0)


	est = mean_reward_estimates
	true = np.array(human_rewards[0])

	true_mean = np.mean(true, axis=0)
	true_std = np.std(true, axis=0)

	est = ((est - np.mean(est, axis=0))/np.std(est, axis=0))*true_std + true_mean
	est_std = np.std(extended_reward_estimates, axis=0) * true_std / np.std(est, axis=0)


	fig, axs = plt.subplots(3,1,sharex=True)

	axs[0].plot(range(1000), est[:,0], '-r', linewidth=2)
	axs[0].fill_between(range(1000), est[:,0] - est_std[:,0], est[:,0] + est_std[:,0], color=(1.0,0.75,0.75))#, alpha=0.25)

	axs[0].plot(range(1000), true[:,0], '--r', linewidth=2)
	axs[0].set_ylim([-3,3])
	axs[0].set_ylabel('Fire', fontsize=14)

	axs[1].plot(range(1000), est[:,1], '-b', linewidth=2)
	axs[1].fill_between(range(1000), est[:,1] - est_std[:,1], est[:,1] + est_std[:,1], color=(0.75,0.75,1.0))#, alpha=0.25)


	axs[1].plot(range(1000), true[:,1], '--b', linewidth=2)
	axs[1].set_ylim([-3,3])
	axs[1].set_ylabel('Triage', fontsize=14)


	axs[2].plot(range(1000), est[:,2], '-g', linewidth=2)
	axs[2].fill_between(range(1000), est[:,2] - est_std[:,2], est[:,2] + est_std[:,2], color=(0.75,1.0,0.75))#, alpha=0.25)

	axs[2].plot(range(1000), true[:,2], '--g', linewidth=2)
	axs[2].set_ylim([-3,3])
	axs[2].set_ylabel('Supply', fontsize=14)
	axs[2].set_xlabel('Time Step', fontsize=14)

	fig.suptitle('Reward Estimate', fontsize=16)
	plt.show()




def get_extended_entropy(time, entropy, max_time=1000):

	cur_entropy = np.zeros((293,5))

	entropies = []

	idx = 0

	for t in range(max_time):
		# Have we gotten to the next reward?
		if idx < len(time) and t == time[idx]:

			cur_entropy = entropy[idx,:]
			idx += 1

		entropies.append(cur_entropy.copy())

	return entropies


def make_perplexity_plot(dataset):
	_, Q = get_reward_parameters(dataset, 'Q')
	times, irlQ = get_reward_parameters(dataset, 'onlineIrl_Q')

	# Use only Q values where times are available
	Qs = [Q[i][times[i]] for i in range(len(Q))]

	# Calculate policies
	Q=np.array(Q)

	full_policy = np.exp(20*Q)/np.sum(np.exp(20*Q), axis=3)[:,:,:,None]
	policy = [np.exp(20*q)/np.sum(np.exp(20*q), axis=2)[:,:,None] for q in Qs]
	irl_policy = [np.exp(20*q)/np.sum(np.exp(20*q), axis=2)[:,:,None] for q in irlQ]

	del Q
	del Qs
	del irlQ

	# Calculate entropy and cross-entropy
	entropy = -np.mean(np.mean(np.sum(full_policy * np.log(full_policy), axis=3), axis=2), axis=0)

	noisy_entropy = []
	for i in range(1,5):
		x=0.25*i
		noisy_policy = (1-x)*full_policy + x*0.2*np.ones(full_policy.shape)
		ne = -np.mean(np.mean(np.sum(noisy_policy*np.log(full_policy), axis=3), axis=2), axis=0)
		noisy_entropy.append(ne)

		del noisy_policy

	del full_policy

	cross_entropy = [irl_policy[i] * np.log(policy[i]) for i in range(len(policy))]
	cross_entropy = np.array([get_extended_entropy(times[i], cross_entropy[i]) for i in range(len(cross_entropy))])
	
	cross_entropy = -np.mean(np.mean(np.sum(cross_entropy, axis=3), axis=2), axis=0)

	return entropy, cross_entropy, noisy_entropy



# Human 2 - 1.5%
# Human 3 - 51.5% / 3.46% / 0.5
# Human 4 - 6.67%
# Human 5 - 27.27% / 1.68%
# Human 6 - 39.47% / 1.64%
# Human 7 - 72.0% / 6.4% 0.1
# Human 8 - 85.29% / 7.09% / 0.05
#dataset = loadFiles('experiments/human_monitoring/')
dataset = loadFiles('experiments/human8/')

make_reward_plot(dataset)
make_kl_plot(dataset)