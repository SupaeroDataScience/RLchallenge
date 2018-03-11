# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 22:13:01 2018

@author: Louis

"""

# Functions used to test during Gcloud training phase.
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

from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten
import graphviz

from collections import deque

def process_screen(x):
    return (255 * resize(rgb2gray(x)[50:, :410], (84, 84))).astype("uint8")

def test_model_G(nb_games, model):
    game = FlappyBird(graphics="fixed") # use "fancy" for full background, random bird color and random pipe color, use "fixed" (default) for black background and constant bird and pipe colors.
    p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=True, display_screen=False)
    p.init()
    reward = 0.0
    
    cumulated = np.zeros((nb_games))
    list_actions = [0,119]
    
    for i in range(nb_games):
        p.reset_game()
        
        while(not p.game_over()):
            state = game.getGameState()
    
            screen_x = process_screen(p.getScreenRGB())
            stacked_x = deque([screen_x, screen_x, screen_x, screen_x], maxlen=4)
            x = np.stack(stacked_x, axis=-1)
            action = list_actions[np.argmax(model.predict(np.expand_dims(x,axis=0)))]
            
            reward = p.act(action)
            
            cumulated[i] = cumulated[i] + reward
    
    avg_score = np.mean(cumulated)
    print('Average : '+ str(avg_score))
    mx_score = np.max(cumulated)
    print('Max : '+ str(mx_score))
    return avg_score, mx_score
