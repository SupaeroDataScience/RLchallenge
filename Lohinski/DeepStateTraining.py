from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np
from random import random, sample
from math import floor

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization

model = Sequential()

model.add(BatchNormalization(input_shape=(8,)))
model.add(Dense(8, kernel_initializer='lecun_uniform', ))
model.add(Activation('relu'))

model.add(Dense(32, kernel_initializer='lecun_uniform'))
model.add(Activation('relu'))
model.add(Dropout(rate=0.2))

model.add(Dense(16, kernel_initializer='lecun_uniform'))
model.add(Activation('relu'))
model.add(Dropout(rate=0.2))

model.add(Dense(2, kernel_initializer='zeros'))
model.add(Activation('tanh'))
model.compile(loss='mean_squared_error', optimizer="adam")

UP = 119
DOWN = 0
actions = [DOWN, UP]
episodes = 10000
epsilon = 1
jumpRate = 0.1
gamma = 0.8
bufferSize = 5
h = 0
replay = []

rewards = np.zeros(episodes, dtype=int)
max_pipes = 0
game = FlappyBird()
env = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=True, display_screen=True)
env.init()
for ep in range(episodes):
    env.reset_game()
    pipes = 0
    state = game.getGameState()
    S = np.array(list(state.values())).reshape(1, len(state))
    while not env.game_over():
        Q = model.predict(S)
        if random() < epsilon:
            A = 1 if random() < jumpRate else 0
        else:
            A = np.argmax(Q)
        r = env.act(actions[A])
        if r == 1.0:
            pipes += 1
        R = 0 if r < 0 else 10 if r > 0 else 1
        rewards[ep] += R
        state = game.getGameState()
        S_ = np.array(list(state.values())).reshape(1, len(state))
        if len(replay) < bufferSize:
            replay.append((S, A, R, S_))
        else:
            h = (h + 1) if h < bufferSize - 1 else 0
            miniBatch = sample(replay, bufferSize)
            X_train = []
            Y_train = []
            for m in miniBatch:
                s, a, r, s_ = m
                q = model.predict(s)
                q_ = model.predict(s_)
                maxQ = np.max(q_)
                if r != 0:
                    r += gamma * maxQ
                y = np.array(q)
                y[0][a] = r
                X_train.append(np.array(s).reshape(8, ))
                Y_train.append(np.array(y).reshape(2, ))
            X_train = np.array(X_train)
            Y_train = np.array(Y_train)
            model.fit(X_train, Y_train, batch_size=bufferSize, epochs=1, verbose=False)
            S = S_
    if pipes > max_pipes:
        max_pipes = pipes
        print('Max passed pipes: {}'.format(max_pipes))
    epsilon -= 1 / episodes
    if ep % floor(episodes / 10 if episodes <= 10000 else episodes / 100) == 0:
        print('Games: {}/{}'.format(ep, episodes))
    if (ep + 1) % 100 == 0:
        model.save('model_{}_epochs.h5'.format(episodes))

max_score = np.max(rewards)
print('Max score is : {}'.format(max_score))
print('Max number of pipes passed : {}'.format(max_pipes))
