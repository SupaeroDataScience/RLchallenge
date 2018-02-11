import random
from skimage.color import rgb2gray
from skimage.transform import resize
import pygame
import numpy as np
import network as nt
from network import UP, ACTIONS

X = []
model = nt.DQN.load_dqn("350_000_ite")

def FlappyPolicy(state, screen):
    s = nt.process_screen(screen)
    if len(X) < 4:
        X.append(s)
        return UP
    else:
        X.append(s)
        X.pop(0)
        x = np.stack(X, axis=-1)
        y = nt.greedy_action(model, x)
        action = ACTIONS[y]
        print(action)
        return action
