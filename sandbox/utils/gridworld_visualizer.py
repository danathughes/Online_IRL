## GridworldVisualizer
##
##

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib import cm

class GridworldVisualizer:
	"""
	A window to visualize the current state of the gridworld environment
	"""

	def __init__(self, env, state_feature_map):
		"""
		"""

		self.env = env
		self.state_feature_map = state_feature_map

		# Start the graph to plot
		plt.ion()

		self.data = np.zeros(self.env.size)
		

		self.bg_color = 'white'
		self.agent_color = 'black'
		self.target_1_color = 'blue'
		self.target_2_color = 'green'
		self.target_3_color = 'red'

		self.cmap = matplotlib.colors.ListedColormap([self.bg_color, self.agent_color, self.target_1_color, self.target_2_color, self.target_3_color])
		self.bounds = [-0.5,0.5,1.5,2.5,3.5,4.5]
		self.norm = matplotlib.colors.BoundaryNorm(self.bounds, self.cmap.N)

		self.graph = plt.imshow(self.data, cmap=self.cmap, norm=self.norm, interpolation='nearest')
		plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False)

		plt.show()

		self.update_graph()


	def update_graph(self):
		"""
		"""

		self.data = np.zeros(self.env.size)

		# Add agent
		agent_x, agent_y = self.env.current_state

		self.data[agent_x, agent_y] = 1.0

		# Add targets
		for i in range(self.state_feature_map.num_features):
			target_states = self.state_feature_map.get_states_with_feature(i)
			for x,y in target_states:
				self.data[x,y] = 2.0 + i

		# Place the data in the graph
		self.graph.set_data(self.data.T)
		plt.pause(0.1)
