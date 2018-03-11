#A soul in tension, that's learning to fly...
#Condition-grounded but determined to try

from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np
import random
from IPython.display import clear_output
from datetime import datetime
import os
import glob
import shutil
import time
import matplotlib.pyplot as plt

start_time = time.time() 

############
#Constantes#
############
epochs = 7200 #Nombre de parties à jouer, obtenu par essai-erreur pour avoir une moyenne de 15 pour 100 parties jouées
gamma = 0.8 # discount factor
alpha = 0.6 
prob = 0.9 
epsilon = 0 #Par essai-erreur 
print_delay = 100 #Utilisé pour l'affichage


###########
#Fonctions#
###########


#L'utilisation du vecteur de sortie total du jeu en entrée d'un réseau de neurones ayant été un échec, je me suis tournée vers la version "easy". 
#Je considère que la version précédente n'a pas fonctionné parce que le vecteur d'entrée de mon réseau de neurones était trop grand
#Pour pallier ce problème je simplifie ce vecteur d'entrée en ne gardant que les paramètres qui me semblent utiles
#(modifiés pour obtenir des grandeurs "physiques" comme la distance horizontale au prochain tuyau) 
#De cette façon on peut conserver les états plutôt que de passer par un réseau de neurones.
#Intuitivement on a besoin de 2 informations : la distance de Flappy bird au prochain tuyau et sa vitesse verticale.
#Ces informations sont stockées sous la forme d'une chaîne de caractères.
def getstate(jeu):
    
    a = list(jeu.getGameState().values())
    
    #Calcul de la distance verticale entre Flappy bird et le prochain tuyau :
    #On a "normalisé" pour avoir des valeurs positives et un nombre d'états raisonnables (par essai-erreur)
    v_pos = np.int(np.round((a[0]-a[3]+512)/10))
    
    #Même chose avec la distance horizontale
    h_pos = np.int(np.round((a[2]+20)/10))
    
    #Vitesse verticale de l'oiseau
    speed = np.int(np.round(a[1]))
    
    #On stocke ces valeurs dans Q à l'aide d'une chaîne de caractères
    S = 'v'+np.str(v_pos)+'h'+np.str(h_pos)+'s'+np.str(speed)
    
    return S

Q = {}

#Les récompenses sont directement les sorties de p.act(action):
reward_dict = { "positive": 1, "negative": 0.0, "tick": 0, "loss": -1000.0, "win": 0.0}

# Variables de suivi
scores = []
average = []
games = []
    
#Initialisation du jeu
jeu = FlappyBird()
p = PLE(jeu, fps=30, frame_skip=1, num_steps=1, force_fps=True, display_screen=True, reward_values = reward_dict)


#################
##Apprentissage##
#################

for i in range(epochs):
    p.reset_game()
    S = getstate(jeu)
    while(not jeu.game_over()):
        if S not in Q:
            Q[S] = {None : 0, 119 : 0}
        if (np.random.random() < epsilon): # Phase d'exploration
            if np.random.random() < prob: 
                action = None
            else:
                action = 119
        else: # Phase d'exploitation
            if Q[S][119] > Q[S][None]:
                action = 119
            else:
                action = None
        r = p.act(action)
        
        if r == 0: #Flappy bird s'est maintenu en l'air, ce qui est déjà un bon point
            reward = 1
        elif r == 1: #Flappy bird a franchi un tuyau ce qui est encore mieux
            reward = 100
        else: #Flappy bird est tombé ou a touché un tuyau (ou a touché le plafond)
            reward = -1000
        
        new_S = getstate(jeu)        
        if new_S not in Q:
            Q[new_S] = {None : 0, 119 : 0}
        maxQ = np.maximum(Q[new_S][None],Q[new_S][119])        
        Q[S][action] = Q[S][action] + alpha*(reward + gamma*maxQ - Q[S][action])      
        S = new_S
        
    scores.append(jeu.getScore()-reward_dict["loss"])
    
    if epsilon > 0.2:
        epsilon -= (1.0/epochs)
    
    ##########################    
    #Suivi de l'apprentissage#
    ##########################

    #Le suivi se fait à une fréquence déterminée par print_delay
    #On sauvegarde également la matrice Q à cette fréquence (choix arbitraire)
    if ((i+1)%print_delay) == 0:
        print("")
        print("Game #: %s" % (i+1,))
        print("Score moyen sur les ",print_delay," derniers essais =", np.mean(scores))
        elapsed_time = time.time() - start_time
        printime = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        print("Temps écoulé : ", printime)
        average.append(np.mean(scores))
        games.append(i+1)
        np.save('Q',Q)
        print('Save done avec moyenne: ' + np.str(np.mean(scores)))
        scores = []