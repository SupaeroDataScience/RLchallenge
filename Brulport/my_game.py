# You're not allowed to change this file
from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np
import pygame
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import resize
from time import time
import pickle

game = FlappyBird()
p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=True, display_screen=True)
# Note: if you want to see you agent act in real time, set force_fps to False. But don't use this setting for learning, just for display purposes.

p.init()
reward = 0.0
UP = 119
nb_games = 10
cumulated = np.zeros((nb_games))

states = []  # s, a, r ,s'


def gamer_policy():
    timeout = time() + 0.04
    action = 0
    while time() < timeout:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYUP:
                action = UP
    return action


def process_screen(x):
    return 256 * resize(rgb2gray(x), (102, 100))[18:, :84]


actions = []
images = []
rewards = []
game_id = []
for i in range(nb_games):
    p.reset_game()

    while (not p.game_over()):
        state = game.getGameState()
        screen = p.getScreenRGB()
        images.append(process_screen(screen))

        game_id.append(i)

        action = gamer_policy()
        actions.append(action)

        reward = p.act(action)
        rewards.append(reward)
        cumulated[i] = cumulated[i] + reward


average_score = np.mean(cumulated)
max_score = np.max(cumulated)

filehander = open("images.pickle", "wb")
pickle.dump(images, filehander)
filehander.close()

for i, a in enumerate(actions):
    if a == UP:
        actions[i] = 1

filehander = open("actions.pickle", "wb")
pickle.dump(actions, filehander)
filehander.close()

filehander = open("rewards.pickle", "wb")
pickle.dump(rewards, filehander)
filehander.close()

filehander = open("game.pickle", "wb")
pickle.dump(game_id, filehander)
filehander.close()
