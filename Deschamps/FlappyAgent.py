import numpy as np
actions = [None, 119]
Q = np.load('Q_good.npy').item()


# def getstate(state):
    
#     state_list = list(state.values())
#     v_pos = state_list[0]-state_list[4]+512 # Différence de position verticale entre l'oiseau et le bas du prochain tuyau
#                           #+512 (hauteur totale en pixels) pour ne jamais être négatif
#     h_pos = state_list[2] #Distance de l'oiseau au prochain tuyau
#     return v_pos, h_pos

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
