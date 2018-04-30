import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression as linear
from scipy import stats
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

# Checks integerness
def is_int(s):
	try: 
		i = int(s)
		return True, i
	except ValueError:
		return False, -1.0

# Divider for output sections
def divider(n = 100):
	print('{}'.format('-'*n))

radius_of_earth = 6371

class Traffic:

	def __init__(self, file = '../data/MT_cleaned.csv'):
		self.df = pd.read_csv(file, low_memory = False)
		self.num_stops = len(self.df)
		self.state = self.df.id[0][:2]	 
		self.years = []
		self.model = linear()
		self.avg_m_years = []
		self.max_county = ''
		self.max_county_area = 0
		self.num_arrests = len(self.df[self.df.is_arrested == True])
		self.dui = sum([type(v) == str and 'DUI' in v for v in self.df.violation])/float(self.num_stops)  
		self.hr_stop = []
		self.oos_table = []
		self.oos = 1 
	 
	def set_out_of_state(self):
		oos = self.df[self.df.out_of_state == True].is_arrested
		num_oos = len(oos)
		oos_arr = sum([x == True for x in oos])
		st = self.df[self.df.out_of_state == False].is_arrested
		num_st = len(st)
		st_arr = sum([x == True for x in st])
		p_oos = oos_arr/float(num_oos)
		p_st = st_arr/float(num_st)
		self.oos_table.append([oos_arr, num_oos - oos_arr])
		self.oos_table.append([st_arr, num_st - st_arr])
		self.oos = p_oos/p_st

	def set_year_of_manufacture_model(self):
		self.years, y_prev, index = [], 0, []
		for i, date in enumerate(self.df.stop_date):
			if type(date) == str:
				y = int(date[:4])
				if y > y_prev:
					self.years.append(y)
					index.append(i)
					y_prev = y
		m_years = []
		for i, year in enumerate(self.years):
			if i < len(self.years) - 1:
				end = index[i+1]
			else:
				end = self.num_stops
			for y in self.df.vehicle_year[index[i]: end]:
					b, yr = is_int(y)
					if b is True:
						m_years.append(yr)
			self.avg_m_years.append(np.nanmean(m_years))
			m_years = []
		self.years = np.array(self.years, dtype = 'float64')
		self.avg_m_years = np.array(self.avg_m_years, dtype = 'float64')
		self.model.fit(np.array(self.years).reshape(-1, 1), self.avg_m_years)

	def p_value(self):
		y_pred = self.model.predict(self.years.reshape(-1, 1))
		total_ss = sum((self.avg_m_years - y_pred)**2)
		num_s = len(self.years)
		ones = np.ones(num_s)
		x = self.years - np.mean(self.years)*ones
		sxx = np.dot(x,x)
		se = np.sqrt(total_ss/((num_s - 2)*sxx))
		t = self.model.coef_[0]/se
		return 2*(1-stats.t.cdf(np.abs(t), num_s - 2))

	def area(self, county):
			coords = self.df[self.df.county_name == county].iloc[:,[22, 23]].fillna(0)
			coords = coords[(coords.lat > 25) & (coords.lon < -85)]
			dlat, dlon = np.std(coords.lat), np.std(coords.lon)
			a, b = dlat*radius_of_earth*np.pi/180, dlon*radius_of_earth*np.pi/180
			return np.pi*a*b

	@timer
	def set_max_county(self):
		counties = list(set(self.df.county_name))
		counties.sort()
		areas = []
		for county in counties:
			if str(county) != 'nan':
				area = self.area(county)
				#print county, area
				areas.append(area)
		i = np.argmax(areas)
		self.max_county_area = areas[i]
		self.max_county = counties[i]

	@timer
	def set_hr_stop(self):
		for i, t in enumerate(self.df.stop_time):
			if type(t) == str:
				t = t.split(':')
				if int(t[1]) >= 30:
					t = (int(t[0]) + 1) % 24
				else:
					t = int(t[0])
				self.df.at[i, 'stop_time'] = str(t)
		for hr in range(24):
			self.hr_stop.append(len(self.df[self.df.stop_time == str(hr)]))

def chi_sq(table):
	table = np.array(table)
	total = float(np.sum(table))
	table0 = np.zeros((2, 2))
	rowsum, colsum = [], []
	for i in range(2):
		rowsum.append(sum(table[i]))
		colsum.append(sum(table[:,i]))
	for i in range(2):
		for j in range(2):
			table0[i][j] = rowsum[i]*colsum[j]/total
	return np.sum((table - table0)**2/table0)

def diff_stops(tr1, tr2):
	tr1.set_hr_stop()
	tr2.set_hr_stop()
	hs = np.array(tr1.hr_stop) + np.array(tr2.hr_stop)
	return max(hs) - min(hs)

trM = Traffic('../data/MT_cleaned.csv')
trV = Traffic('../data/VT_cleaned.csv')


divider()
# Male fraction
print('Proportion of traffic stops in Montana involving male drivers: {}'.\
	format(len(trM.df[trM.df.driver_gender == 'M'])/float(trM.num_stops)))
divider()
# Out of state
trM.set_out_of_state()
print('Number of times an out of state vehicle is more likely to get stopped in Montana: {}'\
	.format(trM.oos))
divider()
# Chi squared
print('Chi-squared statistic for out of state and state populations in Montana: {}'\
	.format(chi_sq(trM.oos_table)))
divider()
# Speeding
print('Proportion of speeding violation in Montana: {}'\
	.format(sum([type(v) == str and 'Speeding' in v for v in trM.df.violation])/float(trM.num_stops)))
divider()
# DUI
print('DUI Proportions, Montana vs Vermont: {}'.format(trM.dui/trV.dui))
divider()
# Average year of manufacture
trM.set_year_of_manufacture_model()
print('Predicted mean of manufacture year for 2020 in Montana: {}'.format(trM.model.predict([[2020]])[0]))
divider()
# p-value
print('p-value of regression: {}'.format(trM.p_value()))
divider()
# Difference between max, min stops hour-wise
print('Difference between number of stops for hours with most and least stops: {}'\
	.format(diff_stops(trM, trV)))
divider()
# Largest county by area
trM.set_max_county()
print('Area of the largest county in Montana in sq km: {}'.format(trM.max_county_area))
divider()