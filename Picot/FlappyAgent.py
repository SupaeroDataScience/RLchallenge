#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 15:10:47 2018

@author: ericpicot
"""

import numpy as np
from ple.games.flappybird import FlappyBird
from ple import PLE
import pickle
from tempfile import TemporaryFile
import matplotlib.pyplot as plt
import random
from IPython.display import clear_output
epochs = 100
alpha = 0.4
gamma = 0.95
terminal_state_size = 9
cumulated = np.zeros((epochs))

# Par souci de simplicité, je modifie le nom des états
a='player_y'
b='player_vel'
c='next_pipe_dist_to_player'
d='next_pipe_top_y'
e='next_pipe_bottom_y'
f='next_next_pipe_dist_to_player'
g='next_next_pipe_top_y'
h='next_next_pipe_bottom_y'


# On charge le file dans lequel se trouve la matrice de la fonction de valeur Q.
f_myfile = open('Q_function.pickle', 'rb')
Q_function = pickle.load(f_myfile)  # variables come out in the order you put them in
f_myfile.close()



def bool_to_act(integer): # Permet de transformer l'indice argMax de la fonction de valeur en action
    if integer == 1:
        return 119
    else:
        return None
    
def act_to_bool(Action): # Permet de transformer une action du jeu en indice {0,1}  pour la fonction de valeur
    if Action == 119:
        return 1
    else:
        return 0


# Je choisi de ne me servir que des états "position verticale", "vitesse" et les distances horizontales et verticales au prochain tuyau
liste_etat=[a,b,c,d]

def reduce_state(state): # Fonction réduisant la dimension de l'espace de travail, ne gardant que 4 états dont on diminue la taille : 
    
    s=[int(state[Etat_partiel]) for  Etat_partiel in liste_etat] # on ne garde que les états mentionnés ci-dessus
    
    output_state = []
    output_state.append(int(s[2]/4)) # distance horizontal au Pipe
    output_state.append(int(s[1]/2)) # vitesse du oiseau
    output_state.append(int((s[3]-s[0])/8)) #position relative, si positive, l'oiseau est au dessus
    
    return output_state

def FlappyPolicy(state, screen):
    

        RS= reduce_state(state) #réduction de l'état
        qval = Q_function[RS[0]][RS[1]][RS[2]] #On accède à l'état demandé
    
        if (qval[0]==qval[1]): # Comme il est possible que lors de l'apprentissage, tous les états n'aient pas été "découvert", et que lors de l'initialisation
                              #  de Q_function qval[0]=qval[1], alors si tel est le cas, je choisi de rien faire car il est plus risqué de flappé que de ne rien faire
            Action = None
            
        else: #choose best action from Q(s,a) values
            Action = bool_to_act(qval.argmax())
        return Action

