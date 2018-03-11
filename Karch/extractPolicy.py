import pickle
from time import time

import numpy as np
import pygame
from ple import PLE
from ple.games.flappybird import FlappyBird
from skimage.color import rgb2gray
from skimage.transform import resize


def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)


def process_screen(x):
    '''
    Transform the rgb image into grayscale image and resize it to 80x80.
    The transformation takes out the ground and the part of the screen which is behind flappybird
    :param np.array x: screen to process.
    :return: processed screen.
    :rtype: np.array
    '''
    return 256 * resize(rgb2gray(x), (102, 100))[18:102, :84]


def gamer_policy():
    timeout = time() + 0.05
    while time() < timeout:
        for event in pygame.event.get():
            if event.type == pygame.KEYUP:
                return UP
    return None


game = FlappyBird(graphics="fixed")
p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=True, display_screen=True)
# Note: if you want to see you agent act in real time, set force_fps to False. But don't use this setting for learning, just for display purposes.s
p.init()
reward = 0.0

ACTIONS = p.getActionSet()
UP = ACTIONS[0]
NO_UP = ACTIONS[1]

nb_games = 10
cumulated = np.zeros((nb_games))

screens = []
memory = []

for i in range(nb_games):
    p.reset_game()
    while (not p.game_over()):
        screen = p.getScreenRGB()
        action = gamer_policy()
        reward = p.act(action)
        if reward == 0: reward = 0.1
        if reward == -5: reward = -1
        if action == None: action = 0
        screens.append(screen)
        memory.append((process_screen(screen), action, reward))
    cumulated[i] = cumulated[i] + reward

save_object(memory, 'human_policy/human_replay.pkl')
