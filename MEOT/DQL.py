# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 10:50:17 2018

@author: Louis
"""

import os

os.environ['SDL_VIDEODRIVER'] = 'dummy'
from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np
from FlappyAgent import FlappyPolicy

import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.exposure import rescale_intensity

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
import graphviz

from collections import deque

def process_screen(x):
    return rescale_intensity(256*resize(rgb2gray(x), (72,128))[10:,:100], in_range=(0,255))


#%% Network Definition*
dqn = Sequential()
#1st layer
dqn.add(Conv2D(filters=16, kernel_size=(8,8), strides=4, activation="relu", input_shape=(62,100,4)))
#2nd layer
dqn.add(Conv2D(filters=32, kernel_size=(4,4), strides=2, activation="relu"))
dqn.add(Flatten())
#3rd layer
dqn.add(Dense(units=256, activation="relu"))
#output layer
dqn.add(Dense(units=2, activation="linear"))

dqn.compile(optimizer="rmsprop", loss="mean_squared_error")

#%% Training Fonctions

def epsilon(step):
    if step<50000:
        return 1.-step*(0.95/50000)
    return .05

def clip_reward(r):
    rr=1
    if (r==0):
        rr=0.1
    if (r==1):
        rr= 1
    if r<0:
        rr=-1
    return rr

def greedy_action(network, x):
    Q = network.predict(np.array([x]))
    return np.argmax(Q)

def MCeval(network, trials, length, gamma):
    p.reset_game()
    scores = np.zeros((trials))
    for i in range(trials):
        screen_x = process_screen(p.reset_game())
        stacked_x = deque([screen_x, screen_x, screen_x, screen_x], maxlen=4)
        x = np.stack(stacked_x, axis=-1)
        for t in range(length):
            a = greedy_action(network, x)
             
            r = p.act(a)
            r = clip_reward(r)
            raw_screen_y = p.getScreenRGB()
            screen_y = process_screen(raw_screen_y)
            scores[i] = scores[i] + gamma**t * r
            if p.game_over()==True:
                # restart episode
                screen_x = process_screen(p.reset_game())
                deque([screen_x, screen_x, screen_x, screen_x], maxlen=4)
                x = np.stack(stacked_x, axis=-1)
            else:
                # keep going
                screen_x = screen_y
                stacked_x.append(screen_x)
                x = np.stack(stacked_x, axis=-1)
    return np.mean(scores)

#%% Memory_buffer
# A class for the replay memory


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
        self.rewards = np.zeros((length,1), dtype=np.int8) # rewards
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


#%% Training Episode
# initialize state and replay memory  
game = FlappyBird(graphics="fixed") # use "fancy" for full background, random bird color and random pipe color, use "fixed" (default) for black background and constant bird and pipe colors.
p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=True, display_screen='store_false')
# Note: if you want to see you agent act in real time, set force_fps to False. But don't use this setting for learning, just for display purposes.

p.init()

total_steps = 200000
replay_memory_size = 20000
mini_batch_size = 32
gamma = 0.95


p.reset_game()
screen_x = process_screen(p.getScreenRGB())
stacked_x = deque([screen_x, screen_x, screen_x, screen_x], maxlen=4)
x = np.stack(stacked_x, axis=-1)
replay_memory = MemoryBuffer(replay_memory_size, (62,100), (1,))
# initial state for evaluation
evaluation_period = 10
Xtest = np.array([x])
nb_epochs = total_steps // evaluation_period
epoch=-1
scoreQ = np.zeros((nb_epochs))
scoreMC = np.zeros((nb_epochs))


# Deep Q-learning with experience replay
for step in range(200000):
    
    if (step%50000==0):
        dqn.save_weights('TrainG1_'+str(step)+'.h5')
        dqn.save('TrainG1.h5')
    
    # evaluation
#    if(step%10 == 0):
#        epoch = epoch+1
#        # evaluation of initial state
#        scoreQ[epoch] = np.mean(dqn.predict(Xtest).max(1))
#        # roll-out evaluation
#        scoreMC[epoch] = MCeval(network=dqn, trials=20, length=700, gamma=gamma)
    # action selection
    print(step)
    if np.random.rand() < epsilon(step):
        if np.random.randint(10)<=1:
            print('randomup')
            a = 1
        else :
            print('randomdown')
            a = 0
    else:
        print('greedy')
        a = greedy_action(dqn, x)
        if a==1 : 
            print('greedyup')
    # step

    r=p.act(a*p.getActionSet()[0])
    raw_screen_y = p.getScreenRGB()
    
    r = clip_reward(r)
    d=p.game_over()
    
    screen_y = process_screen(raw_screen_y)
    replay_memory.append(screen_x, a, r, screen_y, d)
    
    # train
    if step>mini_batch_size:
        X,A,R,Y,D = replay_memory.minibatch(mini_batch_size)
        QY = dqn.predict(Y)
        QYmax = QY.max(1).reshape((mini_batch_size,1))
        update = R + gamma * (1-D) * QYmax
        QX = dqn.predict(X)
        QX[np.arange(mini_batch_size), A.ravel()] = update.ravel()
        dqn.train_on_batch(x=X, y=QX)
        
    # prepare next transition
    if d==True:
        # restart episode
        p.reset_game()
        screen_x = process_screen(p.getScreenRGB())
        stacked_x = deque([screen_x, screen_x, screen_x, screen_x], maxlen=4)
        x = np.stack(stacked_x, axis=-1)
    else:
        
        # keep going
        screen_x = screen_y
        stacked_x.append(screen_x)
        x = np.stack(stacked_x, axis=-1)


dqn.save_weights('TrainG1_max.h5')
dqn.save('TrainG1_max.h5')