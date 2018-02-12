import numpy as np
actions = [None, 119]
Q = np.load('Q_good.npy').item()

def getstate(state):
	a = list(state.values())
	v_pos = np.int(np.round((a[0]-a[3]+512)/10))
	h_pos = np.int(np.round((a[2]+20)/10))
	speed = np.int(np.round(a[1]/2))
	S = 'h'+np.str(h_pos)+'v'+np.str(v_pos)+'s'+np.str(speed)
	return S

def FlappyPolicy(state, screen):

	S = getstate(state)
	if S not in Q:
		action = None
	elif Q[S][119]>Q[S][None]:
		action = 119
	else:
		action = None
        
	return action
