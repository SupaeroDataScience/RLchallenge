from ple.games.flappybird import FlappyBird
from ple import PLE

from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Flatten

import numpy as np

list_actions = [None,119]
nb_save = 4
save_pipe_center = []

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

dqn.load_weights("test_final.dqf") # Permet de charger le résultat précédent.

def FlappyPolicy(state, screen):
    return NemoPolicy(state)
    

def DoriPolicy(state):
    next_pip_center = (state['next_pipe_bottom_y']+state['next_pipe_top_y'])/2
    if state['player_y'] > next_pip_center:
        return list_actions[1]
    else:
        return list_actions[0]

def NemoPolicy(state):
    global save_pipe_center, nb_save
    
    next_pipe_center = (state['next_pipe_bottom_y']+state['next_pipe_top_y'])/2
    
    if len(save_pipe_center)==0:
        save_pipe_center = [next_pipe_center for i in range(nb_save+1)]
    else:
        save_pipe_center.append(next_pipe_center)
        
    if state['player_y'] > save_pipe_center.pop(0):
        return list_actions[1]
    else:
        return list_actions[0]
    
    
def greedy_action(network, state, batchSize): #Cherche la meilleure action prédite.
    qval = network.predict(state.reshape(1,len(state)), batch_size=batchSize)
    qval_av_action = [-9999]*2
    for ac in range(0,2):
        qval_av_action[ac] = qval[0][ac]
    action = (np.argmax(qval_av_action))
    return action
    
def Reinforcement_learning_policy(state):
    return list_actions(greedy_action(dqn, state, batchSize))