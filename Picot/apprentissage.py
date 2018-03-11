#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 14:28:49 2018

@author: ericpicot
"""
import numpy as np
from ple.games.flappybird import FlappyBird
from ple import PLE
from tempfile import TemporaryFile
import matplotlib.pyplot as plt
import random
from IPython.display import clear_output
import pickle



# Initialisation "légèrement aléatoire" de Q_functiON. la valeur vaut soit 1 soit 0 pour chaque Q(s,a)
Q_function = np.random.randint(low=0,high = 2, size =(int(288/4),30,int(512/8),2))



# Import de la fonction calculée lors de la phase d'apprentissage
f_myfile = open('Q_functiondepuisleparadisFalse.pickle', 'rb')
Q_function = pickle.load(f_myfile)
f_myfile.close()


# Par souci de simplicité, je modifie le nom des états
a='player_y'
b='player_vel'
c='next_pipe_dist_to_player'
d='next_pipe_top_y'
e='next_pipe_bottom_y'
f='next_next_pipe_dist_to_player'
g='next_next_pipe_top_y'
h='next_next_pipe_bottom_y'


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

    


epochs = 10000 # Nombre de match d'entrainement. En réalité, je me suis arreté à environ 2800 en apprentissage avec Force_fps = True (voir ci-dessous)
alpha = 0.4
gamma = 0.95
terminal_state_size = 9 #Nombre d'états que je considère comme critiquement dangereux. Ceux-ci ayant menés à la mort de Flappy doivent donc être fortement pénalisés.
# A l'inverse, je considère que tout état où Flappy est toujours vivant mérite d'être récompensé.
cumulated = np.zeros((epochs))
epsilon_act = 0.05

### Apprentissage ___________________________________________________

#Philosophie: Il est délicat d'interpreter en direct la qualité d'une décision. Donc j'ai choisi de travailler en off-line. L'ensemble de la partie est enregistrée
#dans une variable 'partie'. A chaque fois que flappy est vivant il le reward vaut 1. Il vaut 6 lorsqu'il passe un tuyau. 
#
#En revanche, à la fin de la partie un certain nombre d'états (ici 9), est pénalisé. J'ai choisi 9 car c'est approximativement une limite à partir de laquelle il est possible de 
#réagir pour éviter le drame.
#
#La fonction Q_fuction est initalisée légèrement aléatoire. Le modèle apprend très vite (en 1h environ) puis commence à avoir de bonne performance.
#J'ai donc choisi de conserver 5% d'aléatoire pour qu'il continue de découvrir des états et se tromper afin d'éviter des matchs à rallonge (comme 1491, son record). De plus, cela 
# permet de mettre légèrement en difficulté le système et de lui faire apprendre comment réagir plus rapidement
#
#Toutes les 500 parties, j'enregistre via Pickle ma Q_function


game = FlappyBird()
p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=True, display_screen=True)
p.init()
for k in range(epochs):
    
    
    print("Game #: %s" % (k,))
    if k !=0: # Pour toutes les parties sauf la première
    
        for last in range(9):
            
            partie[-last-1][2]=-1000 #Les 9 derniers états se voient fortement pénalisés à - 1000
        
        for ea in partie:
            
            old_state, action, reward, futur_state = ea
# Mise à jour Offline 
# Formule du cours:  Q(s,a) = Q(s,a) + alpha*(R + gamma* max(Q(s',a)) - Q(s,a))          
            Q_function[old_state[0]][old_state[1]][old_state[2]][action] =  Q_function[old_state[0]][old_state[1]][old_state[2]][action] + alpha*(reward+gamma*max(Q_function[futur_state[0]][futur_state[1]][futur_state[2]])- Q_function[old_state[0]][old_state[1]][old_state[2]][action] )
            
        
        partie=[]
        p.reset_game()
        state=game.getGameState()
        RS= reduce_state(state)
        

    else: # Pour la première partie


        partie=[]
        p.reset_game()
        state=game.getGameState()
        RS= reduce_state(state)

    while(not p.game_over()):
    

        epsilon = np.random.uniform(0,101) 
        if epsilon > epsilon_act: #Q-greedy
            
            qval = Q_function[RS[0]][RS[1]][RS[2]]
            if (qval[0]==qval[1]):#Comme il est possible que lors de l'apprentissage, tous les états n'aient pas été "découvert", et que lors de l'initialisation
                                  #  de Q_function qval[0]=qval[1], alors si tel est le cas, je choisi de rien faire car il est plus risqué de flappé que de ne rie
                Action = None
                
            else: #choose best action from Q(s,a) values
                Action = bool_to_act(qval.argmax())
        else: #epsilon-greedy
            
            Action = bool_to_act(np.random.randint(0,2))
        
        r=5*p.act(Action)+1    # fonction reward vaut 1 si le jeu n'est pas fini et 6 si un tuyau a été franchi
        cumulated[k] = cumulated[k] + (r-1)/5
    
    
        new_state = game.getGameState()
        ns=reduce_state(new_state)
        partie.append([RS,act_to_bool(Action),r,ns])    #On enregistre l'état, l'action, la récompense et le futur état
    
        RS=ns # mise à jour de l'état
    
    
    # save the model every 500 epochs
    if k%500 == 0:
        print("saving model")
        f_myfile = open('Q_function_avecrandom.pickle', 'wb')
        pickle.dump(Q_function, f_myfile)
        f_myfile.close()
 
