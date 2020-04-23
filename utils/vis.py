import matplotlib.pyplot as plt 
from matplotlib import colors
from matplotlib import cm
import numpy as np


class TimeSeriesVisualizer:
	"""
	A visualizer for time series data
	"""

	def __init__(self, colors, max_time=None, y_max = None, title=None, include_variance=False):
		"""
		"""

		self.colors = colors
		self.num_series = len(self.colors)
		self.time = []
		self.data = [[] for _ in range(self.num_series)]
		self.max_time = max_time
		self.y_max = y_max
		self.title = title

		self.axes=None
		self.plot_handles = []
		self.variance_handles = []

		self.include_variance = include_variance
		if self.include_variance:
			self.variance_data_upper = [[] for _ in range(self.num_series)]
			self.variance_data_lower = [[] for _ in range(self.num_series)]


	def build(self, axes):
		
		# Create a plot
		self.axes = axes

		for i in range(self.num_series):
			self.plot_handles.append(self.axes.plot([], [], linewidth=2, color=self.colors[i]))

			if self.include_variance:
				lower_handle = self.axes.plot([], [], linewidth=1, color=self.colors[i], linestyle="--")
				upper_handle = self.axes.plot([], [], linewidth=1, color=self.colors[i], linestyle="--")
				self.variance_handles.append([lower_handle,upper_handle])





		if self.max_time is not None:
			self.axes.set_xlim(0, self.max_time)
		if self.y_max is not None:
			self.axes.set_ylim(-self.y_max, self.y_max)
		if self.title is not None:
			self.axes.set_title(self.title)


	def update(self):
		
		for i in range(self.num_series):
			self.plot_handles[i][0].set_xdata(self.time)
			self.plot_handles[i][0].set_ydata(self.data[i])		

			if self.include_variance:
				self.variance_handles[i][0][0].set_xdata(self.time)
				self.variance_handles[i][0][0].set_ydata(self.variance_data_lower[i])
				self.variance_handles[i][1][0].set_xdata(self.time)
				self.variance_handles[i][1][0].set_ydata(self.variance_data_upper[i])



	def add(self, time, value, variance=None):
		"""
		"""

		self.time.append(time)
		for i in range(self.num_series):
			self.data[i].append(value[i])		

			if self.include_variance and variance is not None:
				self.variance_data_lower[i].append(value[i] - variance[i])
				self.variance_data_upper[i].append(value[i] + variance[i])



class AgentVisualizer:
	"""
	A visualizer for the position of agents in the gridworld.
	"""

	def __init__(self, color, initial_position=(-1,-1)):
		"""
		Create the visualizer
		"""

		self.color = color
		self.position = initial_position

		self.agent_circle = plt.Circle(self.position, 0.4, facecolor=color, edgecolor='black', linewidth=2)


	def build(self, axes):
		"""
		"""

		axes.add_artist(self.agent_circle)


	def updatePosition(self, position):
		"""
		"""

		x,y=position

		self.agent_circle.set_center((y,x))



class GridworldValueVisualizer:
	"""
	A visualizer for the value of each state in a gridworld.  Assumes the MDP
	has a Gridworld environment or equivalent.
	"""

	def __init__(self, mdp, v_max=10.0, title=None):
		"""
		Create a visualizer
		"""

		# Matplotlib stuff, some to be built later
		self.axes = None
		self.value_data = None
		self.title = title

		# Get the needed information from the MDP
		self.shape = mdp.environment.shape
		self.stateSpace = mdp.stateSpace

		self.v_max = v_max


	def build(self, axes, r_max = None):
		"""
		Initialize a grid and extract the handler for the data
		"""

		# Store the axes and create a handler for the data
		self.axes = axes

		self.data = self.axes.imshow(np.zeros(self.shape), cmap=cm.get_cmap("seismic"), norm=colors.Normalize(-self.v_max,self.v_max))

		if self.title is not None:
			self.axes.set_title(self.title)

		# Draw gridlines
		self.axes.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
		self.axes.set_xticks([])
		self.axes.set_yticks([])


	def valueToGrid(self, data):
		"""
		Convert the values to entries in the visualized grid data
		"""

		gridValues = np.zeros(self.shape)

		for state in self.stateSpace:
			x,y = state
			gridValues[x,y] = data[self.stateSpace(state)]

		return gridValues


	def updateValue(self, V, Q):
		"""
		Update the value funciton
		"""

		self.data.set_array(-self.valueToGrid(V))


	def update(self):
		"""
		"""

		pass



class GridworldVisualizer:
	"""
	"""

	def __init__(self, mdp, tasks, task_colors, blocked=None, title=None):
		"""
		Create a visualizer
		"""

		# Matplotlib stuff, some to be built later
		self.axes = None
		self.title = title

		# Get the needed information from the MDP
		self.shape = mdp.environment.shape
		self.stateSpace = mdp.stateSpace

		# Decorations
		self.decorations = []


		# Create a colormap and boundaries for the grid to use
		# Add black for blocked regions, and white for non-blocked, non-task
		# regions
		self.colors = ['black','white'] + list(task_colors)
		self.color_map = colors.ListedColormap(self.colors)

		# Boundaries are going to be the task number +/- 0.5, as well as bounds
		# for blocked regions (-2.5 to -1.5) and clear regions (-1.5 to -0.5)
		self.bounds = [-2.5 + i for i in range(len(self.colors) + 1)]
		self.norm = colors.BoundaryNorm(self.bounds, self.color_map.N)

		self.blocked = blocked
		self.tasks = tasks


	def add(self, decoration):
		"""
		"""

		self.decorations.append(decoration)


	def build(self, axes):
		"""
		Initialize a grid and extract the handler for the data
		"""

		# Store the axes and create a handler for the data
		self.axes = axes

		# Draw the map and hang on to the handler
		self.map_data = self.axes.imshow(np.zeros(self.shape), cmap=self.color_map, norm=self.norm)

		if self.title is not None:
			self.axes.set_title(self.title)

		# Draw gridlines
		self.axes.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
		self.axes.set_xticks([])
		self.axes.set_yticks([])

		# Draw the initial data
		self.update()

		# Build the decorations
		for decoration in self.decorations:
			decoration.build(self.axes)


	def update(self):
		"""

		"""

		# Reset each cell to indicate empty (-1)
		map_values = np.zeros(self.shape) - 1

		# Add any blocked cells
		if self.blocked is not None:
			map_values = map_values - self.blocked

		# Set tasks
		if self.tasks is not None:
			for loc, task in self.tasks.tasks.items():
				x,y=loc
				map_values[x,y] += task + 1

		# Update the grid data
		self.map_data.set_array(map_values)





class MatplotVisualizer:
	"""
	"""

	def __init__(self):
		"""
		grid_shape - tuple describing the size of the grid (W,H)
		task_colors - list of colors to use for each task (numbered from 0 to N-1)
		blocked - array-like of blocked cells, must be same shape as grid
		tasks - dictionary of task locations to task number
		"""


		self.visualizers = []

		# Create the figure 
		plt.ion()
		self.figure = plt.figure()

		# Do the initial plot
		plt.show()


	def add(self, visualizer, position):
		"""
		Build the visualizer in the given position
		"""

		axes = self.figure.add_subplot(position)
		visualizer.build(axes)
		self.visualizers.append(visualizer)


	def close(self):
		plt.close(self.figure)


	def redraw(self):
		"""
		Re-render the scene
		"""

		for visualizer in self.visualizers:
			visualizer.update()
		self.figure.canvas.draw()
		self.figure.canvas.flush_events()

		