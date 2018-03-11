#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 14:53:45 2018

@author: romainlefebvre94
"""
import numpy as np
import pickle

# Le nombre total d'états étant fortement élevé, il convient d'une part, de
# privilégier certains d'entre eux (voir plus bas) et d'autre part, de réduire
# le champ des états possibles en les classifiant

# maillage vertical de la position relative de Flappy par rapport au tuyau
h_max = 220
h_min = -400
nb_states_vert = 9

def state_vert(player_y):
    k = 0
    while (player_y - h_min > k * (h_max - h_min) / nb_states_vert):
            k = k + 1
    return k

# maillage horizontal de la position relative de Flappy par rapport au tuyau
w = 288
nb_states_pipe = 5

def state_pipe(next_pipe_dist_to_player):
    k = 0
    while (next_pipe_dist_to_player > k * w / nb_states_pipe):
            k = k + 1
    return k

# classes de vitesses afin de balayer les différents états sans considérer
# l'intégralité des vitesses possibles
v_max = 10
v_min = -16
nb_states_vel = 8

def state_vel(player_vel):
    k = 0
    while (player_vel - v_min > k * (v_max - v_min) / nb_states_vel):
            k = k + 1
    return k
   

# conversion des actions 0 ou 1 en actions compréhensibles par le module
# FlappyBird() : 0 --> None = pas d'action, 1 --> 119 = haut
def binary_act(a):
    return None if(a == 0) else 119

# On charge Q calculé dans FlappyQLearning
f_myfile = open('donneesQ.pickle','rb')
Q1 = pickle.load(f_myfile)
f_myfile.close()


# Fonction renvoyant une politique gloutonne par rapport à Q
def greedyPolicy(Q):
    pi = np.zeros((nb_states_vert+1,nb_states_pipe+1,nb_states_vel+1),dtype=np.int)
    for x1 in range(nb_states_vert+1):
        for x2 in range(nb_states_pipe+1):
            for x3 in range(nb_states_vel+1):
                pi[x1][x2][x3] = np.argmax(Q[x1][x2][x3][:])
    return pi

# Politique gloutonne par rapport à Q1 (variable tampon contenant Qql)
pi = greedyPolicy(Q1)

# Fonction d'application de la politique
def FlappyPolicy(state, screen):
    s1 = state_vert(state['next_pipe_top_y']-state['player_y'])
    s2 = state_pipe(state['next_pipe_dist_to_player'])
    s3 = state_vel(state['player_vel'])
    action = binary_act(pi[s1][s2][s3])
    return action