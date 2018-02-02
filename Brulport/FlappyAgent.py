import random
from skimage.color import rgb2gray
from skimage.transform import resize
import pygame
import numpy as np
import network

UP = 119
ACTIONS = dict()
ACTIONS[0] = 0
ACTIONS[1] = UP


def process_screen(x):
    return 256 * resize(rgb2gray(x), (102, 100))[18:, :84]


def greedy_action(network, x):
    Q = network.predict(np.array([x]))
    print(Q[0])
    return np.argmax(Q[0])


X = []
model = network.DQN.load_dqn("second")


def FlappyPolicy(state, screen):
    s = process_screen(screen)
    if len(X) < 4:
        X.append(s)
        if len(X) == 0:
            return UP
        else:
            return UP
    else:
        X.append(s)
        eject = X.pop(0)
        x = np.stack(X, axis=-1)
        y = greedy_action(model, x)
        action = ACTIONS[y]
        return action
