# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 14:55:30 2018

@author: Jules Dubois
"""
from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np
import pickle
import os

game = FlappyBird()
p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=True, display_screen=False)

p.init()

# Booléen pour choisir si on veut entraîner ou tester
training = False

# Fonction de test qui choisit l'action à effectuer en ce basant sur la politique pi fournie
def FlappyPolicy(state,screen):
    if pi[int((412 + state['player_y'] - state['next_pipe_bottom_y'])/6), int(state['next_pipe_dist_to_player']/4), int((16+state['player_vel'])/2) ] == 0:
        return None
    else:
        return 119


# Fonction d'entrainement qui implémente un Q-learning
def Qlearn():
    
    # Données qui permettent de diviser verticalement, horizontalement et selon l'échelle de vitesse
    # les variables d'état et donc de réduire l'espace d'états
    heightScale = 6
    widthScale = 4
    velScale = 2
    
    # Tailles respectives des espaces d'états et d'actions
    states_space_size = (int(924/heightScale), int(288/widthScale), int(28/velScale))
    action_space_size = 2
    
    # Initialisation du nombre de jeux effectués lors de l'entrainement
    jeux = 1
    
    # gamma est-il vraiment utile ici ? On veut tout le temps être autant vivant. On le fixe donc à 1
    gamma = 1
    alpha = 0.3 # Après différents essais, de bons résultats sont obtenus avec cette valeur de alpha
    
    # On initialise Q à 0
    Qql = np.zeros((states_space_size[0], states_space_size[1], states_space_size[2], action_space_size), dtype=np.dtype('float'))
    count = np.zeros((states_space_size[0], states_space_size[1], states_space_size[2], action_space_size), dtype=np.dtype('int16'))
    epsilon = 0 #Epsilon ne sera finalement pas utilisé ici
    
    # Initialisation du premier jeu d'entrainement
    p.reset_game()
    state = game.getGameState()
    x = [int((412 + state['player_y'] - state['next_pipe_bottom_y'])/heightScale), int(state['next_pipe_dist_to_player']/widthScale), int((16+state['player_vel'])/velScale)]
    SA = [] # états actions stockés pour chaque jeu, pour choisir les rewards plus facilement
    
    # Variable comptant le nombre de pas de temps désirés pour l'entrainement
    # On fixe le nombre de pas de temps à la place du nombre de jeux, 
    # on choisit ainsi plus facilement le temps d'entrainement
    t=0
    
    while (t < 10000000):
        t+=1
        # Ici l'exploration est réalisée par le caractère aléatoire de l'emplacement des tubes,
        # on utilise donc un opérateur constamment greedy qui explorera tous les états à force de jouer
        a = epsilon_greedy(Qql,x,epsilon, action_space_size)
        r = p.act(zeroUnToAction(a))
        state = game.getGameState()
        y = [int((412 + state['player_y'] - state['next_pipe_bottom_y'])/heightScale), int(state['next_pipe_dist_to_player']/widthScale), int((16+state['player_vel'])/velScale)]
        SA.append([x[0],x[1],x[2],a])
        count[x[0], x[1], x[2], a] += 1
        
        #Lorsqu'un jeu se termine, on met à jour Q avec les valeurs contenues dans SA temporairement pour ce jeu
        if p.game_over()==True:
            jeux += 1
            print("jeu numéro :" ,jeux-1)
            #Mise à jour de Q
            for i in range(len(SA)-1):
                # On choisit de pénaliser fortement les 10 derniers mouvements ayant conduit à l'échec
                # Les précédents, ayant permis de rester en vie, sont récompensés par un reward de 1
                if (i < (len(SA) - 10)):
                    r=1
                    Qql[SA[i][0], SA[i][1], SA[i][2], SA[i][3]] += alpha * (r+gamma*np.max(Qql[SA[i+1][0],SA[i+1][1], SA[i+1][2],:])-Qql[SA[i][0], SA[i][1], SA[i][2],SA[i][3]])
                else:
                    r=-1000
                    Qql[SA[i][0], SA[i][1], SA[i][2], SA[i][3]] += alpha * (r+gamma*np.max(Qql[SA[i+1][0],SA[i+1][1], SA[i+1][2],:])-Qql[SA[i][0], SA[i][1], SA[i][2],SA[i][3]])
            # On ré-initialise SA pour qu'il soit pour le prochain jeu
            SA = []
            #Ré-initilisation du jeu
            p.reset_game()
            state = game.getGameState()
            x = [int((412 + state['player_y'] - state['next_pipe_bottom_y'])/heightScale), int(state['next_pipe_dist_to_player']/widthScale), int((16+state['player_vel'])/velScale)]
        else:
            x=y
    
    print(jeux, " jeux")
    print("Saving model")
    # La politique est sauvegardée dans un .pickle permettant d'y accéder facilement par la suite
    os.chdir('C:\\Users\\Jules\\Documents\\Supaéro\\4A\\SDD\\Reinforcement\\FlappyBird\\RLchallenge\\Dubois')
    saveFile = open('pi.pickle', 'wb')
    pickle.dump(greedyQpolicy(Qql), saveFile)
    saveFile.close()
    
    return greedyQpolicy(Qql)

# Fonction choisissant l'action greedy. Epsilon peut être modifié pour introduire un aspect exploratoire
def epsilon_greedy(Q, s, epsilon, action_space_size):
    a = np.argmax(Q[s[0],s[1],s[2],:])
    if(np.random.rand()<=epsilon): # Action à peu près aléatoire: la probabilité de ne rien faire est légèrement plus grande pour que l'oiseau ne s'emplafonne pas systématiquement
        a = np.random.randint(-3,2)
        if a < 0:
            a = 0
    return a

# Fonction utilitaire pour obtenir une action
def zeroUnToAction(a):
    if a == 0:
        return a
    elif a >= 1:
        return 119
    else:
        print('got a problem')

# Fonction retournant la politique greedy à partir du Q final déterminé après entrainement
def greedyQpolicy(Q):
    pol = np.zeros((Q.shape[0], Q.shape[1], Q.shape[2]),dtype=np.dtype('int8'))
    for s1 in range(Q.shape[0]):
        for s2 in range(Q.shape[1]):
            for s3 in range(Q.shape[2]):
                pol[s1,s2,s3] = zeroUnToAction(np.argmax(Q[s1,s2,s3,:]))
    return pol

# Fonction permettant de lire le .pickle
def readFile():
    saveFile = open('pi.pickle', 'rb')
    pol = pickle.load(saveFile) 
    saveFile.close()
    return pol


## 

if training:
    pi = Qlearn()
else:
    pi = readFile()



