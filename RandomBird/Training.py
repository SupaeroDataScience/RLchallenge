# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 15:24:01 2018

Fichier d'apprentissage du Flappy Bird

@author: Gaspard Berthelin
"""

from ple.games.flappybird import FlappyBird
from ple import PLE

from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Flatten

import numpy as np

## Initialisation du jeu : 
game = FlappyBird(graphics="fixed")
p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=True, display_screen=True)
p.init()
list_actions=[None,119]

## Initialisation du réseau de neurones :
batchSize = 256 # mini batch size
## couches du réseau de neurone : plusieurs couches ne permettent pas de converger plus rapidement par expérience.
dqn = Sequential()
# 1st layer
#dqn.add(Dense(units=112, init='lecun_uniform', activation="relu", input_shape=(8,)))
# 2nd layer
dqn.add(Dense(units=500, init='lecun_uniform', activation="relu", input_shape=(8,)))
# 3rd layer
#dqn.add(Dense(units=112, init='lecun_uniform', activation="relu", input_shape=(8,)))
# output layer
dqn.add(Dense(units=2, init='lecun_uniform', activation="linear"))
dqn.compile(loss="mean_squared_error", optimizer=optimizers.Adam(1e-4))

dqn.load_weights("final.dqf") # Permet de charger le résultat précédent.

def process_state(state): #Adaptaion de l'état pour en faire une entrée du réseau de neurone.
    return np.array(list(state.values()))

def epsilon(step,total): # Définie la probabilité de choisir le cas aléatoire.
    p_max = 0.4
    p_min = 0.2
    d = 0.0
    x = step/total
    if x < d:
        return p_max
    elif x > (1-d):
        return p_min
    return (p_min-p_max) * (x-d)/(1-2*d) + p_max

def clip_reward(r): #Récompense sur l'évaluation des résultats (permet de donner le nombre de tuyaux passés)
    rr=0
    if r>0:
        rr=1
    if r<0:
        rr=0
    return train4000score13.0

def training_reward(r): #Récompense du training : fort handicap sur le game over.
    rr=0
    if r>0:
        rr=1
    if r<0:
        rr=-1000
    return rr

def random_action(state): # Pour accélérer la convergence, j'ai essayé un état aléatoire plus cohérent. Ce ne fut pas concluant à long terme.
    if state[0] > (state[3]+state[4])/2.0 :
        return 1
    else :
        return 0        
    
def greedy_action(network, state, batchSize): #Cherche la meilleure action prédite.
    qval = network.predict(state.reshape(1,len(state)), batch_size=batchSize)
    qval_av_action = [-9999]*2
    for ac in range(0,2):
        qval_av_action[ac] = qval[0][ac]
    action = (np.argmax(qval_av_action))
    return action

def MCeval(network, trials): #Evalue la qualité du réseau jusqu'à présent.
    scores = np.zeros((trials))
    for i in range(trials):
        p.reset_game()
        while not(p.game_over()):
            state = game.getGameState()
            state = process_state(state)
            action = greedy_action(network, state, batchSize)
            action = list_actions[action]
            reward = p.act(action)
            reward = clip_reward(reward)
            state = game.getGameState()
            state = process_state(state)
            scores[i] = scores[i] +  reward             
    return np.sum(scores)


## Training loop :
total_games = 15000      #nombre de parties jouées pour l'entraînement.
evaluation_period = 1000 #Tout les ... évalue la qualité du réseau.
gamma = 0.99             #Permet de déifinir la récompense update.
step_game = 0            #indice du nombre de parties jouées.
while (step_game < total_games):
    p.reset_game()       #réinitialisation du jeu
    state = game.getGameState()
    state = process_state(state)
    rand_sum = 0
    greedy_sum = 0
    tuyau_passe = 0
    while(not game.game_over()):
        
        if (np.random.random() < epsilon(step_game,total_games)): 
            #Exploration
            rand_sum = rand_sum + 1
            #action = random_action(state)
            action = np.random.choice([0,1])
        else: 
            #On suit le résultat du réseau de neurone.
            greedy_sum = greedy_sum + 1
            action = greedy_action(dqn, state, batchSize)
        
        #Résultat de l'action :
        reward = p.act(list_actions[action])
        reward = training_reward(reward)
        if reward > 0:
            tuyau_passe = tuyau_passe + 1
        
        new_state = game.getGameState()
        new_state = process_state(new_state)
        
        terminal = game.game_over()   
        
        ## Update du réseau de neurone : 
        newQ = dqn.predict(new_state.reshape(1,len(state)), batch_size=batchSize)
        maxQ = np.max(newQ)
        y = np.zeros((1,2))
        y = dqn.predict(new_state.reshape(1,len(state)), batch_size=batchSize)
        
        update = reward + gamma * (1-terminal) * maxQ

        y[0][action] = update
        
        dqn.fit(state.reshape(1, len(state)), y, batch_size=batchSize, nb_epoch=3, verbose=0)
        state = new_state
        
    print("game %d : score = %d / prop_alea = %s pourcents" % (step_game ,tuyau_passe,round(rand_sum/(rand_sum+greedy_sum)*100)))
    if step_game%evaluation_period == 0:
        mcval = MCeval(dqn,50)
        print('eval_MC=',MCeval(dqn,50))
        dqn.save("train"  + str(step_game) + ".dqf")
    
    step_game = step_game + 1
dqn.save("final.dqf")       