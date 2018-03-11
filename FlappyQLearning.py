#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 14:53:45 2018

@author: romainlefebvre94
"""
from ple.games.flappybird import FlappyBird
from ple import PLE
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

# déclaration des variables utiles
Qql = np.zeros((nb_states_vert+1,nb_states_pipe+1,nb_states_vel+1,2))
gamma = 1 # on veut autant être en vie plus tard que maintenant
alpha = 0.4
epsilon = 1
g = 0

##############################################################################
##                                                                          ##
##                          ZONE D'APPRENTISSAGE                            ##
##                                                                          ##
##############################################################################

# Chargement du jeu et initialisation de celui-ci
game = FlappyBird()

p = PLE(game, fps = 30, display_screen = False, force_fps = True)
p.init()

# Initialisation des paramètres de stockage
max_steps = 100000
T = dict()

p.reset_game()
t0 = 0
cumul = 0

# état initial de Flappy
state0 = game.getGameState()
    
x1 = state_vert(state0['next_pipe_top_y']-state0['player_y'])
x2 = state_pipe(state0['next_pipe_dist_to_player'])
x3 = state_vel(state0['player_vel'])

# boucle d'apprentissage
for t in range(max_steps):

    # Action maximisant Qql pour l'état x1, x2, x3    
    a = np.argmax(Qql[x1][x2][x3][:])
    
    # traduction de l'action pour le jeu    
    action = binary_act(a)
    
    # réalisation de l'action et récupération de la récompense associée    
    r = p.act(action)
    # le jeu est-il fini ?
    d = p.game_over()
    
    # enregistrement de l'action pour une itération ultérieure
    T[t] = [x1,x2,x3,a]
    
    # calcul du cumul des gain
    cumul += r
    
    # nouvel état atteint après cette action        
    state = game.getGameState()
    
    y1 = state_vert(state['next_pipe_top_y']-state['player_y'])
    y2 = state_pipe(state['next_pipe_dist_to_player'])
    y3 = state_vel(state['player_vel'])
    
    # Si le jeu est fini, on sanctionne les 9 dernières actions et récompense
    # toutes les précédentes    
    if d==True:
        g = g + 1
        print('g',g, '| t',t, '| c',cumul)
        
        for m in range(t0,t-1):
            # état considéré
            x1 , x2, x3, a = T[m]
            # état ultérieur à l'état considéré
            y2 , y2, y3, inutile = T[m+1]
            
            if (m < t-9):
                # récompense
                Qql[x1][x2][x3][a] += alpha * (1+gamma*np.max(Qql[y1][y2][y3][:])-Qql[x1][x2][x3][a])
            else:
                # sanction
                Qql[x1][x2][x3][a] += alpha * (-1000+gamma*np.max(Qql[y1][y2][y3][:])-Qql[x1][x2][x3][a])
        
        # on relance le jeu et réinitialise les paramètres        
        p.reset_game()
        t0 = t
        cumul = 0
        
        state0 = game.getGameState()
    
        x1 = state_vert(state0['next_pipe_top_y']-state0['player_y'])
        x2 = state_pipe(state0['next_pipe_dist_to_player'])
        x3 = state_vel(state0['player_vel'])
        
    else:
        x1, x2, x3 = y1, y2, y3


# on stocke le Qql dans une variable tampon au cas où...
Q1 = Qql

# enregistrement de Q
f_myfile = open('donneesQ.pickle','wb')
pickle.dump(Q1,f_myfile)
f_myfile.close()

##############################################################################
##############################################################################

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


# Zone de test
game = FlappyBird()

p2 = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=True, display_screen=False)

p2.init()
reward = 0.0

nb_games = 100
cumulated = np.zeros((nb_games))

for i in range(nb_games):
    p2.reset_game()
    print(i)
    while(not p2.game_over()):
        state = game.getGameState()
        screen = p2.getScreenRGB()
        action=FlappyPolicy(state, screen) ### Your job is to define this function.
        
        reward = p2.act(action)
        cumulated[i] = cumulated[i] + reward

average_score = np.mean(cumulated)
max_score = np.max(cumulated)

print(average_score)
print(max_score)