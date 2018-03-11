# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 14:48:23 2018

@author: Louis

"""

#Local test sequence

from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np
from FlappyAgent import FlappyPolicy

import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.exposure import rescale_intensity

from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten
import graphviz

from collections import deque

def process_screen(x):
    return (255 * resize(rgb2gray(x)[50:, :410], (84, 84))).astype("uint8")


def greedy_action(network, x):
    Q = network.predict(np.array([x]))
    return np.argmax(Q)


    #%% 
dqn=load_model('TrainG4_19.h5')
game = FlappyBird(graphics="fixed") # use "fancy" for full background, random bird color and random pipe color, use "fixed" (default) for black background and constant bird and pipe colors.
p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=True, display_screen=True)
p.init()
reward = 0.0
list_actions=[0,119]
nb_games = 100
cumulated = np.zeros((nb_games))


for i in range(nb_games):
    p.reset_game()
    
    while(not p.game_over()):
        state = game.getGameState()

        screen_x = process_screen(p.getScreenRGB())
        stacked_x = deque([screen_x, screen_x, screen_x, screen_x], maxlen=4)
        x = np.stack(stacked_x, axis=-1)
        action = list_actions[greedy_action(dqn,x)]
        
        reward = p.act(action)
        cumulated[i] = cumulated[i] + reward

average_score = np.mean(cumulated)
max_score = np.max(cumulated)   