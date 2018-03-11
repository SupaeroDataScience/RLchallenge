# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 11:36:48 2018

@author: Paul
"""



import numpy as np
Q=np.load("my_trained_Q.npy")

def FlappyPolicy(state, screen):

    # Using "state"
    Y = int(288 + (state['next_pipe_top_y'] + state['next_pipe_bottom_y']) * 0.5 - state['player_y'])
    X = int(state['next_pipe_dist_to_player'])
    v = int(state['player_vel'])
                
    action=None
    action = int(np.argmax(Q[Y][X][v][:]))
    if (action == 1): 
            action = 119
          
    return action