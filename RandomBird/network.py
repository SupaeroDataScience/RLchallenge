from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
import parameters
from skimage.color import rgb2gray
from skimage.transform import resize
import numpy as np

def generate_dqn():
	'''
		Generates the deep Q network.
	'''
	dqn = Sequential()
	dqn.add(Conv2D(filters=16, kernel_size=(8,8), strides=4, activation="relu", input_shape=(parameters.IMG_HEIGHT, parameters.IMG_WIDTH,4)))
	dqn.add(Conv2D(filters=32, kernel_size=(4,4), strides=2, activation="relu"))
	dqn.add(Flatten())
	dqn.add(Dense(units=256, activation="relu"))
	dqn.add(Dense(units=2, activation="linear"))
	return dqn

def process_screen(x):
	'''
		Inputs :
			- raw RGB screen
		Outputs :
			- gray resized screen
	'''
	x = x[50:270, :320]
	return 256*resize(rgb2gray(x), (parameters.IMG_HEIGHT, parameters.IMG_WIDTH))

def greedy_action(convnet, x):
	'''
		Inputs :
			- the deep Q network
			- current game screen
		Output :
			- greedy action based on the dqn prediction
	'''
	Q = convnet.predict(np.array([x]))
	return np.argmax(Q)

def epsilon(step):
	'''
		Return the epsilon for the current time step. Epsilon decreases linearly
		from parameters.INITIAL_EPSILON to parameters.FINAL_EPSILON. It starts
		decreasing after parameters.OBSERVE time steps and stops after
		parameters.EXPLORE time steps.
	'''
	if step < parameters.OBSERVE:
	    return 1
	elif step < parameters.EXPLORE:
	    slope = (parameters.FINAL_EPSILON - parameters.INITIAL_EPSILON)/(parameters.EXPLORE - parameters.OBSERVE)
	    intercept = parameters.INITIAL_EPSILON - parameters.OBSERVE*slope
	    return  slope*step + intercept
	else:
	    return parameters.FINAL_EPSILON

def epsilon_greedy_action(convnet, x, step):
	'''
		Return the epsilon greedy action based on the game screen, the dqn
		prediction and the current epsilon value.
	'''
	if np.random.rand() < epsilon(step):
		a = np.random.randint(2)
	else:
		a = greedy_action(convnet, x)
	return a
