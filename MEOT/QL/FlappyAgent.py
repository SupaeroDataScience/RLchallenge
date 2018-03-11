# -*- coding: utf-8 -*-

"""
Created on Wed Jan 24 14:55:41 2018

@author: Louis MEOT
"""
from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np
from random import randint
import math
import pickle

game=FlappyBird()
p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=False, display_screen=True)

p.init()

nb_games = 20
cumulated = np.zeros((nb_games))
f_myfile = open('Q_function600.pickle', 'rb')
Q_function = pickle.load(f_myfile)  # variables come out in the order you put them in
f_myfile.close()

def FlappyPolicy(state, screen):
    a= play_loop(state)
    return a


# Maillage des états 
def observeState(state):      
    y_to_pipe_bottom = state["player_y"] - state["next_pipe_bottom_y"]
    y_cat = 0
    x_cat = 0
    h_max = 412
    h_min = -412
    d_max = 288
    nb_y_cat = 14
    nb_x_cat = 5
    
    while(y_to_pipe_bottom - h_min > (h_max - h_min) * y_cat/nb_y_cat):
        y_cat += 1
    
    while(state["next_pipe_dist_to_player"] > d_max * x_cat/nb_x_cat):
        x_cat += 1
    
    speed_cat = int((state["player_vel"]+16)/2)

    return (x_cat-1,y_cat-1,speed_cat)

    
def epsilon_greedy(Q, s):
    a = np.argmax(Q[s[0]][s[1]][s[2]][:]) # Action optimale avec une proba 1-eps
    return a

def play_loop(state):
    ps = observeState(state)
    action_ind = epsilon_greedy(Q_function,ps) 
    if (action_ind==1):
        action = 119
    else:
        action = None
    return action

        