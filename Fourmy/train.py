import random
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation

from ple.games.flappybird import FlappyBird
from ple import PLE

STATES = [
    'next_next_pipe_top_y', 'next_pipe_top_y', 'next_pipe_bottom_y',
    'next_next_pipe_bottom_y', 'next_next_pipe_dist_to_player',
    'next_pipe_dist_to_player', 'player_y',  'player_vel'
]
ACTIONS = [None, 119]


def state_to_arr(state):
    return np.array([state[feature] for feature in STATES])\
             .reshape(1, len(state))


# Default model used in RL notebook 4
model = Sequential()
model.add(Dense(150, init='lecun_uniform', input_shape=(len(STATES),)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(150, init='lecun_uniform'))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(len(ACTIONS), init='lecun_uniform'))
model.add(Activation('linear'))
model.compile(loss='mse', optimizer="rmsprop")

nb_games = 1000
gamma = 0.5  # discount factor
epsilon = 0.1  # epsilon-greddy

update = 0
alpha = 0.1  # learning rate
experience_replay = True
batch_size = 40  # mini batch size
buff = 80
replay = []  # init vector buffer
buffer_size = 0  # current size of the vector buffer

game = FlappyBird()
p = PLE(game, fps=30, frame_skip=1, num_steps=1,
        force_fps=True, display_screen=True)
# Note: if you want to see you agent act in real time, set force_fps to False.
# But don't use this setting for learning, just for display purposes.

# 1) In s, choose a (GLIE actor)
# 2) Observe r, s′
# 3) Temporal difference:
# delta = r + gamma*maxa′Q(s′,a′)−Q(s,a)δ=r+γmaxa′Q(s′,a′)−Q(s,a)
# 4) Update Q :  Q(s,a) ← Q(s,a) + αδQ(s,a) ← Q(s,a)+αδ
# 5) s <- s′


for i in range(nb_games):
    p.reset_game()
    state = game.getGameState()
    state_arr = state_to_arr(state)

    while not p.game_over():
        # 1) In s, choose a (GLIE actor)
        qval = model.predict(state_arr, batch_size=batch_size)
        if random.random() < epsilon:  # exploration
            action = random.randint(0, len(ACTIONS)-1)
        else:
            action = np.argmax(qval)

        # 2) Observe r, s′
        state = game.getGameState()
        state_arr = state_to_arr(state)
        reward = p.act(ACTIONS[action])
        # 3) Temporal difference
        qval_new = model.predict(state_arr, batch_size=batch_size)
        max_qval = np.max(qval_new)
        delta = reward + gamma*max_qval - qval[0][action]
        y = np.zeros((1, len(ACTIONS)))  # TODO: do we need to keep qval?
        y[0][:] = qval[0][:]
        y[0][action] = qval[0][action] + alpha*delta
        # 4) Update Q
        model.fit(state_arr, y, batch_size=batch_size, nb_epoch=1, verbose=False)

        if reward > 0:
            print('YEAHHHH!')
