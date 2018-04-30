import random
import numpy as np
import time

# Timing wrapper
def timer(func):
	def new_func(*args,**kwargs):
		start = time.time()
		val = func(*args,**kwargs)
		end = time.time()
		print('Time taken by function {} is {} seconds'.format(func.__name__, end-start))
		return val
	return new_func

# Divider for sections in output
def div(n):
	return '-'*n

class Gridwalk:

	def __init__(self, xdim, ydim):
		self.m = xdim
		self.n = ydim
		self.paths = []
		self.devs = []
		self.mean_dev = -1.0
		self.std_dev = -1.0
		print('\n{0}\nGrid-walk for m = {1}, n = {2}\n{0}\n'.format(div(30), xdim, ydim))
	
	def walk(self):
		path = []
		x, y = 0, 0
		while (x + y < self.m + self.n - 1):
			if random.random() < 0.5:
				 x = x + 1
			else:
				y = y +1
			path.append([x, y])
		path.append([self.m, self.n])
		return np.array(path, dtype = 'float32')

	@timer
	def generate_paths(self, num_walks = int(1e4)):
		for i in range(num_walks):
			self.paths.append(self.walk())
		print('{} paths have been generated'.format(num_walks))

	def deviation(self, path):
		d = []
		for x, y in path:
			d.append(abs(x/self.m - y/self.n))
		return max(d)

	@timer
	def stat_dev(self):
		print('Computing deviations ...')
		for path in self.paths:
			self.devs.append(self.deviation(path))
		self.devs = np.array(self.devs, dtype = 'float32')
		self.mean_dev = np.mean(self.devs)
		self.std_dev = np.std(self.devs) 
		
	def cond_prob(self, condition = 0.2, threshold = 0.6):
		events = self.devs[np.where(self.devs > condition)]
		num_threshold_events = sum([e > threshold for e in events]) 
		return num_threshold_events/float(len(events))

	def output(self, num_walks = int(1e4), condition = 0.2, threshold = 0.6):
		self.generate_paths(num_walks)
		self.stat_dev()
		print('\nMean of D = {}\nStandard deviation of D = {}\nConditional probability = {}'\
			.format(self.mean_dev, self.std_dev, self.cond_prob(condition, threshold)))



Gridwalk(11, 7).output()
Gridwalk(23, 31).output()

