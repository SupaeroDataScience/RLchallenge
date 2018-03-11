#import the libraries
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras import optimizers
from keras.models import load_model
from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize

dqn = load_model('dqn_3.h5')
iter = 0
stacked_x = []

def process_screen(x):
	x = x[50:270, :320]
	return 256*resize(rgb2gray(x), (80,80))

def fill_stack(screen):
	deq = deque([screen, screen, screen, screen], maxlen=4)
	return deq
	
def FlappyPolicy(state, screen):
	global stacked_x
	global iter
	global dqn
	
	moves = [1,0]
	iter = iter + 1
	screen_x = process_screen(screen)

	if iter == 1:
		stacked_x = fill_stack(screen_x)
		x = np.stack(stacked_x, axis=-1)
	else:
		stacked_x.append(screen_x)
		x = np.stack(stacked_x, axis=-1)

	QX = dqn.predict(np.array([x]))
	action =  moves[np.argmax(QX)]*119
	return action