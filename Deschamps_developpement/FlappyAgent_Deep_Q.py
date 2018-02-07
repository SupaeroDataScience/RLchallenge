from keras.models import load_model
import numpy as np
actions = [None, 119]
model = load_model('FlappyBird.dqf')

def FlappyPolicy(state, screen):
	state_list = list(state.values())
	
	qval = model.predict(np.array(state_list).reshape(1,len(state_list)), batch_size=1)
	if qval[0][0] > qval[0][1]:
	    action = actions[0]
	else:
	    action = actions[1]
	print(action)
	return action
