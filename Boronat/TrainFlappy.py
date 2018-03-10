from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from skimage.color import rgb2gray
from skimage.transform import resize
import skimage
from keras.optimizers import SGD , Adam
import matplotlib.pyplot as plt
import random
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten

print("TrainFlappy")

def build_dqn():
	dqn = Sequential()
	#1st layer
	dqn.add(Conv2D(filters=16, kernel_size=(8,8), strides=4, activation="relu", input_shape=(80,80,4)))
	#2nd layer
	dqn.add(Conv2D(filters=32, kernel_size=(4,4), strides=2, activation="relu"))
	dqn.add(Flatten())
	#3rd layer
	dqn.add(Dense(units=256, activation="relu"))
	#output layer
	dqn.add(Dense(units=2, activation="linear"))
	adam = Adam(lr=1e-4)
	dqn.compile(optimizer=adam, loss="mean_squared_error")
	return dqn

def policy_eval(p, games, network):
    """
    Equivalnt to MonteCarlo Eval in RL notebook
    """
    list_actions = p.getActionSet()
    cumulated = np.zeros((games))
    for i in range(games):
        stackedframes = deque([np.zeros((80,80)),np.zeros((80,80)),np.zeros((80,80)),np.zeros((80,80))], maxlen=4)
        p.reset_game()
        while(not p.game_over()):
            screen = process_screen(p.getScreenRGB())
            stackedframes.append(screen)
            frameStacked = np.stack(stackedframes, axis=-1)
            action = list_actions[np.argmax(network.predict(np.expand_dims(frameStacked,axis=0)))]
            reward = p.act(action)
            cumulated[i] += reward
    mean_score = np.mean(cumulated)
    max_score = np.max(cumulated)
    return mean_score, max_score

class MemoryBuffer:
    "An experience replay buffer using numpy arrays"
    def __init__(self, length, screen_shape, action_shape):
        self.length = length
        self.screen_shape = screen_shape
        self.action_shape = action_shape
        shape = (length,) + screen_shape
        self.screens_x = np.zeros(shape, dtype=np.uint8) # starting states
        self.screens_y = np.zeros(shape, dtype=np.uint8) # resulting states
        shape = (length,) + action_shape
        self.actions = np.zeros(shape, dtype=np.uint8) # actions
        self.rewards = np.zeros((length,1), dtype=np.uint8) # rewards
        self.terminals = np.zeros((length,1), dtype=np.bool) # true if resulting state is terminal
        self.terminals[-1] = True
        self.index = 0 # points one position past the last inserted element
        self.size = 0 # current size of the buffer
    
    def append(self, screenx, a, r, screeny, d):
        self.screens_x[self.index] = screenx
        #plt.imshow(screenx)
        #plt.show()
        #plt.imshow(self.screens_x[self.index])
        #plt.show()
        self.actions[self.index] = a
        self.rewards[self.index] = r
        self.screens_y[self.index] = screeny
        self.terminals[self.index] = d
        self.index = (self.index+1) % self.length
        self.size = np.min([self.size+1,self.length])
    
    def stacked_frames_x(self, index):
        im_deque = deque(maxlen=4)
        pos = index % self.length
        for i in range(4): # todo
            im = self.screens_x[pos]
            im_deque.appendleft(im)
            test_pos = (pos-1) % self.length
            if self.terminals[test_pos] == False:
                pos = test_pos
        return np.stack(im_deque, axis=-1)
    
    def stacked_frames_y(self, index):
        im_deque = deque(maxlen=4)
        pos = index % self.length
        for i in range(4): # todo
            im = self.screens_y[pos]
            im_deque.appendleft(im)
            test_pos = (pos-1) % self.length
            if self.terminals[test_pos] == False:
                pos = test_pos
        return np.stack(im_deque, axis=-1)
    
    def minibatch(self, size):
        #return np.random.choice(self.data[:self.size], size=sz, replace=False)
        indices = np.random.choice(self.size, size=size, replace=False)
        x = np.zeros((size,)+self.screen_shape+(4,))
        y = np.zeros((size,)+self.screen_shape+(4,))
        for i in range(size):
            x[i] = self.stacked_frames_x(indices[i])
            y[i] = self.stacked_frames_y(indices[i])
        return x, self.actions[indices], self.rewards[indices], y, self.terminals[indices]

# Function to display X and Y from minibatch to visualize what is stored in memory        
def show_X_Y(X,Y):
    temp = np.transpose(X[0],(2,1,0))
    tempy = np.transpose(Y[0],(2,1,0))

    plt.figure(figsize=(30,30))
    plt.subplot(2,4,1)
    plt.imshow(temp[0][:][:])
    plt.subplot(2,4,2)
    plt.imshow(temp[1][:][:])
    plt.subplot(2,4,3)
    plt.imshow(temp[2][:][:])
    plt.subplot(2,4,4)
    plt.imshow(temp[3][:][:])
    plt.title(A[0])
    plt.legend(R[0])

    plt.subplot(2,4,5)
    plt.imshow(tempy[0][:][:])
    plt.subplot(2,4,6)
    plt.imshow(tempy[1][:][:])
    plt.subplot(2,4,7)
    plt.imshow(tempy[2][:][:])
    plt.subplot(2,4,8)
    plt.imshow(tempy[3][:][:])
    plt.show()

def epsilon(step):
    if step<(1e6+99999):
        return (1.-step*9e-7)*0.1
    return 0.001

def clip_reward(r):
    if r!=1:
        rr=0.1 
    else:
        rr=r
    return rr


def greedy_action(network, x):
    Q = network.predict(np.array([x]))
    return np.argmax(Q)

def process_screen(x):
    return 255*resize(rgb2gray(x[60:, 25:310,:]),(80,80))

game = FlappyBird(graphics="fixed") # use "fancy" for full background, random bird color and random pipe color, use "fixed" (default) for black background and constant bird and pipe colors.
p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=True,display_screen=True)
p.init()

print("Test creation model")
model = build_dqn()
print("End creation model")

total_steps = 1000000
replay_memory_size = 1000000
mini_batch_size = 32
gamma = 0.99

 # initialize state and replay memory
p.reset_game()
screen_x = process_screen(p.getScreenRGB())
stacked_x = deque([screen_x, screen_x, screen_x, screen_x], maxlen=4)
x = np.stack(stacked_x, axis=-1)

replay_memory = MemoryBuffer(replay_memory_size, (80, 80), (1,))
evaluation_period = 10000
frame_per_action = 1
score =0

list_actions = p.getActionSet()

# Deep Q-learning with experience replay
for step in range(total_steps*frame_per_action):

    if (step%frame_per_action == 0):

    	if np.random.rand() < epsilon(step):
            a_test= np.random.randint(0,5)
            if (a_test==1):
            	a=1
            else:
            	a=0
        else:
            a = greedy_action(model,x)

        action = list_actions[a]
        r = clip_reward(p.act(action))
        raw_screen_y = p.getScreenRGB()
        screen_y = process_screen(raw_screen_y)

        #Display image to check if cropped correctly and consistency with game
        #plt.imshow(screen_y)
        #plt.show()

        d = p.game_over()

        replay_memory.append(screen_x, a, r, screen_y, d)
        
        if step>evaluation_period*frame_per_action:
            X,A,R,Y,D = replay_memory.minibatch(mini_batch_size)
            QY = model.predict(Y)
            QYmax = QY.max(1).reshape((mini_batch_size,1))
            update = R + gamma * (1-D) * QYmax
            QX = model.predict(X)
            QX[np.arange(mini_batch_size), A.ravel()] = update.ravel()
            model.train_on_batch(x=X, y=QX)

        if d==True:
            p.reset_game()
            #score = 0
            screen_x = process_screen(p.getScreenRGB())
            stacked_x = deque([screen_x, screen_x, screen_x, screen_x], maxlen=4)
            x = np.stack(stacked_x, axis=-1)
        else:
            # keep going
            screen_x = screen_y
            stacked_x.append(screen_x)
            x = np.stack(stacked_x, axis=-1)

    if (step%50000*frame_per_action == 0):
    	print(str(step)+str(policy_eval(p,10,model)))
        model.save("/media/martin/disque_stockage/Hugo/RLchallenge-master/RandomBird/FlappyModel_"+str(step))