import numpy as np
import _pickle as cPickle

def discrete_state(state):
	x = str(int(round(state['next_pipe_dist_to_player']/20)))
	y = str(int(round((state['player_y'] - state['next_pipe_bottom_y'])/20)))
	v = str(int(round(state['player_vel'])))
	return x+"-"+y+"-"+v
flag_dict=False
Q= dict()

def FlappyPolicy(state, screen):
	action=None
	global flag_dict
	global Q
	
	if not flag_dict:
		Q = cPickle.load(open("Qql",'rb'))
		flag_dict=False
	s=discrete_state(state)

	if s in Q.keys():
		a = np.argmax(Q[s][:])
	else:
		a = 0

	if a==0:
		action=0
	else:
		action=119
	
	return action


