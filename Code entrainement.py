# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 11:01:57 2018

@author: Paul
"""

from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np
import matplotlib.pyplot as plt

game = FlappyBird()
p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=True, display_screen=False)

p.init()
reward = 0.0
nb_games = 10000   
cumulated = np.zeros((nb_games))

# Paramètres du modèle
r_plus = 1
r_moins = -100
alpha = 0.04 

# On retient tous les features de la partie
X_ante = np.zeros((37))
Y_ante = np.zeros((37))
v_ante = np.zeros((37))
a_ante = np.zeros((37))

# Initialisation de Q - non nécessaire, mais utile pour s'approcher des bons coefficients
Q = np.zeros((576,300, 21, 2))     # Q(Y, X, V, a)
    ## Voler lorsque Y<273, ne pas voler lorsque Y>274
Q[274:575,:,:,0] = 0.1
Q[0:273,:,:,1] = 0.1
    # entre les tuyaux 
Q[:,8,:,1] = 0.2  # au milieu : sauter
Q[265:288,120:144,:,1] = 0.2  # en sortie de tuyau, si position trop basse, sauter
Q[288:335,120:144,:,0] = 0.2   # en sortie de tuyau, si position trop haute, ne pas sauter


for i in range(nb_games):
    p.reset_game()
            
    while(not p.game_over()):
        state = game.getGameState()
        screen = p.getScreenRGB()
        # Calcul des features
        Y = int(288 + (state['next_pipe_top_y'] + state['next_pipe_bottom_y']) * 0.5 - state['player_y'])
        X = int(state['next_pipe_dist_to_player'])
        v = int(state['player_vel'])
        
        # Politique gloutonne, suffisante lorsque Q est bien initialisé
        action = int(np.argmax(Q[Y][X][v][:]))
        if (action == 1): 
            action_value = 119 
        else: action_value=None        
       
        # Sauvegarde des features
        if (i>1):
            for j in range(37-1, 0, -1):
                X_ante[j] = int(X_ante[j-1])
                Y_ante[j] = int(Y_ante[j-1])
                v_ante[j] = int(v_ante[j-1])
                a_ante[j] = int(a_ante[j-1])
            X_ante[0] = int(X)
            Y_ante[0] = int(Y)
            v_ante[0] = int(v)
            a_ante[0] = int(action)
        
        # Récompenses : 
        # Si l'oiseau passe un tuyau : +1 pour les 36 derniers états-actions 
        # (correspond aux 2 derniers tuyaux traversés)
        reward = p.act(action_value)
        my_reward=0
        if (reward==1):
            my_reward = r_plus
            cumulated[i] += 1
            for j in range(1, 37):  
                Q[int(Y_ante[j]),int(X_ante[j]),int(v_ante[j]),int(a_ante[j])] += alpha * (my_reward + np.max(Q[int(Y_ante[j-1]),int(X_ante[j-1]),int(v_ante[j-1]),int(a_ante[j-1])]))
        
        # En cas de collision : -100
        if (reward<0):
            my_reward = r_moins
            if (X==39):
                for j in range(0, 27):
                    Q[int(Y_ante[j]),int(X_ante[j]),int(v_ante[j]),int(a_ante[j])] += alpha * (my_reward + np.max(Q[int(Y_ante[j-1]),int(X_ante[j-1]),int(v_ante[j-1]),int(a_ante[j-1])]))
            else: 
                for j in range(0, 6):
                    Q[int(Y_ante[j]),int(X_ante[j]),int(v_ante[j]),int(a_ante[j])] += alpha * (my_reward + np.max(Q[int(Y_ante[j-1]),int(X_ante[j-1]),int(v_ante[j-1]),int(a_ante[j-1])]))
        

cumulated_resized = cumulated[:i]
m=np.mean(cumulated_resized)
plt.plot(cumulated_resized)
plt.savefig("Rewards.png")

import csv
with open("Average rewards pour Q initialisé, 10000 parties, alpha=0.04, r+=1, r-=-100.csv", "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for ii in range(100):
        writer.writerow([str(ii*100) + "-" + str((ii+1)*100), np.mean(cumulated[100*ii : 100*(ii+1)])])

np.save('trained_Q_from_zero', Q)
    

