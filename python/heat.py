import plotly as py
import pandas as pd 
import time

df = pd.read_csv('../data/stage3.csv')
code = pd.read_csv('https://raw.githubusercontent.com/jasonong/List-of-US-States/master/states.csv')
py.tools.set_credentials_file(username='pinakm9', api_key='rb7LOm9P84uhRI7pF2HZ')

# Timing wrapper
def timer(func):
	def new_func(*args,**kwargs):
		start = time.time()
		val = func(*args,**kwargs)
		end = time.time()
		print('Time taken by function {} is {} seconds'.format(func.__name__, end-start))
		return val
	return new_func

def find_in_state(state, topic, col = 'incident_characteristics'):
	notes = df[df.state == state][col]
	return sum([type(note) == str and topic in note for note in notes])

@timer
def find(topic, col = 'incident_characteristics'):
	counts = []
	for state in code.State:
		counts.append(find_in_state(state, topic))
		print('Finished working in {}, count = {}'.format(state,counts[len(counts)-1] ) )
	return counts

def heatmap(title, topic = 'accident', col = 'incident_characteristics'):
	counts = find(topic, col)
	data = [ dict(\
		type='choropleth',\
		autocolorscale = True,\
		locations = code.Abbreviation,\
		z = counts,\
		locationmode = 'USA-states',\
		marker = dict(\
		    line = dict (\
		        color = 'rgb(255,255,255)',\
		        width = 2\
		    ) )\
		) ]

	layout = dict(
		title = 'US {} by state from 2013-current'.format(title),\
		geo = dict(\
			scope='usa',\
			projection=dict( type='albers usa' ),\
			showlakes = True,\
			lakecolor = 'rgb(255, 255, 255)'),\
		 )
    
	fig = dict( data=data, layout=layout )
	py.plotly.image.save_as( fig, filename='US-{}.png'.format(title) )

heatmap(title = 'accidental shootings', topic = 'Accident')
heatmap(title = 'drive-by shootings', topic = 'Drive-by')