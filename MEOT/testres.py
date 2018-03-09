# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 14:48:23 2018

@author: Louis

"""
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

dqn.load_weights('Train6_100000.h5')
game = FlappyBird(graphics="fixed") # use "fancy" for full background, random bird color and random pipe color, use "fixed" (default) for black background and constant bird and pipe colors.
p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=True, display_screen=True)
p.init()
reward = 0.0

nb_games = 100
cumulated = np.zeros((nb_games))


for i in range(nb_games):
    p.reset_game()
    
    while(not p.game_over()):
        state = game.getGameState()

        screen_x = process_screen(p.getScreenRGB())
        stacked_x = deque([screen_x, screen_x, screen_x, screen_x], maxlen=4)
        x = np.stack(stacked_x, axis=-1)
        action = greedy_action(dqn,x)*119
        
        reward = p.act(action)
        cumulated[i] = cumulated[i] + reward

average_score = np.mean(cumulated)
max_score = np.max(cumulated)   