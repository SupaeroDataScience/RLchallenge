import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

dqn = Sequential()
# 1st layer
dqn.add(Dense(units=500, kernel_initializer='lecun_uniform', activation="relu", input_dim = 8))
# output layer
dqn.add(Dense(units=2, kernel_initializer='lecun_uniform', activation="linear"))

dqn.compile(loss='mse', optimizer=optimizers.Adam(1e-4))

dqn.load_weights("dqn_3_1.dqf")

batchSize = 256
actions = [None, 119]

def process_state(state):
    """ Renvoie l'état sous forme de liste """
    return [state['player_y'], state['player_vel'],
            state['next_pipe_dist_to_player'], state['next_pipe_top_y'], state['next_pipe_bottom_y'],
            state['next_next_pipe_dist_to_player'], state['next_next_pipe_top_y'], state['next_next_pipe_bottom_y']]

def greedy_action(network, state_x):
    """ Renvoie la meilleure action possible """
    Q = network.predict(np.array(state_x).reshape(1, len(state_x)), batch_size=batchSize)
    return np.argmax(Q)

def FlappyPolicy(state, screen):
    state = process_state(state)
    action = greedy_action(dqn, state)
    return actions[action]