#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 10:15:16 2018

This is the code for training flappy bird how to fly
It is a Deep Q-Network train by "screen" variable
To achieve the level obtained with the attached DQN (flappy_brain),
We needed close to 160,000 frames to perform the training, it has been done 
using google cloud engine  


@author: Ilyass_Haloui
"""
from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam
from skimage.color import rgb2gray
from skimage.transform import resize


def createNetwork():
    #Create the CNN with Adam Opt
    DQNF = Sequential()
    DQNF.add(Conv2D(filters=16, kernel_size=(8,8), strides=4, activation="relu", input_shape=(80,80,4)))
    DQNF.add(Conv2D(filters=32, kernel_size=(4,4), strides=2,activation="relu"))
    DQNF.add(Flatten())
    DQNF.add(Dense(units=256, activation="relu"))
    DQNF.add(Dense(units=2, activation="linear"))
    DQNF.compile(optimizer=Adam(lr=1e-4), loss='mean_squared_error')
    print(DQNF.summary())
    return DQNF 


# epsilon=1 on the first 5k steps and then linearly from 0.1 to 0.001 
def epsilon(step):
    if step < 5e3:
        return 1
    elif step < 1e6:
        return (0.1 - 5e3*(1e-3-0.1)/(1e6-5e3)) + step * (1e-3-0.1)/(1e6-5e3)
    else:
        return 1e-3

#We cut screen edges in order to simplify the training
def process_screen(screen):
    return 255*resize(rgb2gray(screen[60:, 25:310,:]),(80,80))

# There is no negative reward !
def clip_reward(r):
    if r!=1:
        rr=0.1 
    else:
        rr=r
    return rr

def greedy_action(network, x):
    Q = network.predict(np.array([x]))
    return np.argmax(Q)

def policy_eval(p, games, network):
    """
    Monte carlo evaluation of the mean score and max score of a 10000 frame episode
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
        self.actions[self.index] = a
        self.rewards[self.index] = r
        self.screens_y[self.index] = screeny
        self.terminals[self.index] = d
        self.index = (self.index+1) % self.length
        self.size = np.min([self.size+1,self.length])

    def stacked_frames_x(self, index):
        im_deque = deque(maxlen=4)
        pos = index % self.length
        for i in range(4):
            im = self.screens_x[pos]
            im_deque.appendleft(im)
            test_pos = (pos-1) % self.length
            if self.terminals[test_pos] == False:
                pos = test_pos
        return np.stack(im_deque, axis=-1)

    def stacked_frames_y(self, index):
        im_deque = deque(maxlen=4)
        pos = index % self.length
        for i in range(4):
            im = self.screens_y[pos]
            im_deque.appendleft(im)
            test_pos = (pos-1) % self.length
            if self.terminals[test_pos] == False:
                pos = test_pos
        return np.stack(im_deque, axis=-1)

    def minibatch(self, size):
        indices = np.random.choice(self.size, size=size, replace=False)
        x = np.zeros((size,)+self.screen_shape+(4,))
        y = np.zeros((size,)+self.screen_shape+(4,))
        for i in range(size):
            x[i] = self.stacked_frames_x(indices[i])
            y[i] = self.stacked_frames_y(indices[i])
        return x, self.actions[indices], self.rewards[indices], y, self.terminals[indices]
    
    

total_steps = 300000
replay_memory_size = 300000
mini_batch_size = 32
gamma = 0.99
eval_period = 10000
nb_epochs = total_steps // eval_period
epoch=-1
stop_training = False


DQNF = createNetwork()


game = FlappyBird(graphics="fixed")
p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=True,
        display_screen=False)

list_actions = p.getActionSet()

p.init()
p.reset_game()

screen_x = process_screen(p.getScreenRGB())
stacked_x = deque([screen_x, screen_x, screen_x, screen_x], maxlen=4)
x = np.stack(stacked_x, axis=-1)

replay_memory = MemoryBuffer(replay_memory_size, screen_x.shape, (1,))

mean_score = np.zeros((nb_epochs))
max_score = np.zeros((nb_epochs))


for step in range(total_steps):
    
    
    # Score evaluation:
    if(step%eval_period == 0 and step>0):
        epoch += 1
        print(f"[ Epoch - periode ] :{ [ (epoch+1) , eval_period] }; ")
        print('Starting score evaluation.. : ')
        DQNF.save('flappy_brain.h5')
        nb_games = 100
        mean_score[epoch], max_score[epoch] = policy_eval(p, nb_games, DQNF)
        print('Score : {}/{} (mean/max)'.format(mean_score[epoch],max_score[epoch]))
        print('Score eval done..')
        
        
        
        if (mean_score[epoch-1] > 30):
            stop_training = True
        
        
        
    if not stop_training:
        if np.random.rand() < epsilon(step):
            a = np.random.randint(0,2)
        else:
            a = greedy_action(DQNF, x)



        r = clip_reward(p.act(list_actions[a]))
        
        
        screen_y = process_screen(p.getScreenRGB())
        
        
        replay_memory.append(screen_x, a, r, screen_y, p.game_over())
        
        
        # train
        if (step > mini_batch_size and step > 10000):
            X,A,R,Y,D = replay_memory.minibatch(mini_batch_size)
            QY = DQNF.predict(Y)
            QYmax = QY.max(1).reshape((mini_batch_size,1))
            update = R + gamma * (1-D) * QYmax
            QX = DQNF.predict(X)
            QX[np.arange(mini_batch_size), A.ravel()] = update.ravel()
            DQNF.train_on_batch(x=X, y=QX)
        # Save regularly:
        if (step > 0 and step % 2500 == 0):
            DQNF.save('flappy_brain.h5')

        #if not terminal stage
        if p.game_over()==True:
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
            
            
    if stop_training:
        break

DQNF.save('flappy_brain_2.h5')


print("Training completed")    



