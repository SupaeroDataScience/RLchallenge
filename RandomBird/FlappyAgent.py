import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.models import load_model
from collections import deque


#Agent intern variables
stacked_x = []
calls = 0
dqn = load_model('Outputs/dqn_700k.h5')
actions = [119, None]

def process_screen(x):
	""" Take FlappyBird RGB screen, convert it into a grayscale image and resize it"""
	x = x[50:270, :320]
	return 256*resize(rgb2gray(x), (80,80))

def FlappyPolicy(state, screen):
	""" Take FlappyBird RGB screen as an input and return the rightaction to do"""

	global stacked_x
	global calls
	global dqn
	global actions

	calls += 1
	screen_x = process_screen(screen)

	if calls == 1:
		stacked_x = deque([screen_x, screen_x, screen_x, screen_x], maxlen=4)
		x = np.stack(stacked_x, axis=-1)
	else:
		stacked_x.append(screen_x)
		x = np.stack(stacked_x, axis=-1)

	Q = dqn.predict(np.array([x]))
	return actions[np.argmax(Q)]
