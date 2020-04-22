try:
	import cPickle as pickle
except:
	import pickle

class Logger:
	"""
	"""

	def __init__(self, path, metadata):
		"""
		path      - the path to store data to
		metadata  - a dictionary of static information about the experiment
		data_keys - a list of keys to store data to at each time step
		"""

		self.path = path
		self.metadata = metadata
		self.data = {}
		self.time = 0


	def setTime(self, time):
		"""
		"""

		self.time = time

		if not self.time in self.data:
			self.data[self.time] = {}


	def log(self, key, value):
		"""
		"""

		# Create a dictionary for the current time if it doesn't
		# exits
		if not self.time in self.data:
			self.data[self.time] = {}

		# Store the value, overwriting any prior results
		self.data[self.time][key] = value


	def save(self):
		"""
		Store the data using pickle
		"""

		log = {'metadata': self.metadata, 'data': self.data}

		with open(self.path, 'wb') as pf:
			pickle.dump(log, pf)