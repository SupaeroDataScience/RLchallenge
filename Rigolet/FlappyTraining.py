# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 14:55:41 2018

@author: Clémence
"""
from ple.games.flappybird import Flapp yBird
from ple import PLE
import numpy as np
from random import randint
import math
import pickle

game=FlappyBird()
p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=True, display_screen=True)

p.init()
reward = 0.0

nb_games = 5000
t = 0
epsilon = 1
cumulated = np.zeros((nb_games))
Qp = np.zeros((5,14,27,2))
Q = np.zeros((5,14,27,2))
nb_states = 0
STATES = []
count = np.zeros((5,14,26,2))

for i in range(nb_games):
    p.reset_game()
    
    while(not p.game_over()):
        t += 1
        if(t == 10000):
            epsilon = epsilon/2
            t = 0
        state = game.getGameState()
        screen = p.getScreenRGB()
        nb_states += 1
        action = FlappyPolicy(state, screen,game,p,epsilon,cumulated,i,count,Q, STATES, nb_states)
    rewardAndUpdateQ(STATES, nb_states, Q)
    nb_states = 0
    STATES = []
    print("GAME NUMBER",i)
    print("Score final = ", cumulated[i])
    
    
average_score = np.mean(cumulated)
max_score = np.max(cumulated)

    
def FlappyPolicyDyn(state,screen):
    print(state)
    if(state["player_y"]>state["next_pipe_bottom_y"]-50):
        return 119
    else: 
        return None
        
def FlappyPolicy(state, screen,game,p,epsilon,cummulated,i,count,Q, STATES, nb_states):
    a= play_loop(state,Q,game,p,epsilon,cumulated,i,count, STATES, nb_states)
    return a

# Maillage des états
def observeState(state,p):      
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

def rewardAndUpdateQ(STATES, nb_states, Q):
    alpha = 0.4
    gamma = 0.9
    
    for i in range(nb_states-1):
        s = STATES[i]
        ns = STATES[i+1]
        if (i<nb_states-9):
            reward = 1
        else:
            reward = -1000
        Q[s[0]][s[1]][s[2]][s[3]] += alpha*(reward+gamma*np.max(Q[ns[0]][ns[1]][ns[2]][:])-Q[s[0]][s[1]][s[2]][s[3]])
      
# Retourne l'action a 
def epsilon_greedy(Q, s, epsilon):
    a = np.argmax(Q[s[0]][s[1]][s[2]][:]) # Action optimale avec une proba 1-eps
    if(np.random.rand()<=epsilon): # random action
        rd = np.random.rand(1) #Seulement deux actions possibles
        if (rd<=0.2):
            a=1
        else:
            a=0
    return a


def play_loop(state,Q,game,p,epsilon,cumulated,i,count, STATES, nb_states):
    ps = observeState(state,p)
    action_ind = epsilon_greedy(Q,ps,epsilon) 
    STATES.append([ps[0],ps[1],ps[2],action_ind])
    if (action_ind==1):
        action = 119
    else:
        action = None
    game_reward = p.act(action)  #Fait l'action
    state = game.getGameState() # Nouvel état
    new_state = observeState(state,p)
    cumulated[i] += game_reward
    return action

########### SAuvegarde de la matrice entraînée
print("saving model")
f_myfile = open('Q_function_3566_ite.pickle', 'wb')
pickle.dump(Q, f_myfile)
f_myfile.close()

# Read from file
f_myfile = open('Q_function_3566_ite.pickle', 'rb')
Q_function = pickle.load(f_myfile)  # variables come out in the order you put them in
f_myfile.close()
        