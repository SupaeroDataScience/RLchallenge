# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 11:39:06 2018

@author: Louis
"""



from ple.games.flappybird import FlappyBird
from ple import PLE
from PIL import Image
import numpy as np
from FlappyAgent import FlappyPolicy
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import resize




def process_screen(x):
    return 256*resize(rgb2gray(x), (72,128))[10:,:100]


#%% 
game = FlappyBird(graphics="fixed") # use "fancy" for full background, random bird color and random pipe color, use "fixed" (default) for black background and constant bird and pipe colors.
p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=False, display_screen=True)
# Note: if you want to see you agent act in real time, set force_fps to False. But don't use this setting for learning, just for display purposes.

p.init()

    #%%
reward = 0.0

nb_games = 100
cumulated = np.zeros((nb_games))


for i in range(nb_games):
    p.reset_game()
    
    while(not p.game_over()):
        state = game.getGameState()
        screen = process_screen(p.getScreenRGB())
        

        
        action=FlappyPolicy(state, screen) ### Your job is to define this function.
        
        
        
        reward = p.act(action)
        cumulated[i] = cumulated[i] + reward

average_score = np.mean(cumulated)
max_score = np.max(cumulated)


#####----------
