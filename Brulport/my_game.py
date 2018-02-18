# You're not allowed to change this file
from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np
import pygame
from skimage.color import rgb2gray
from skimage.transform import resize
from time import time
import pickle
from network import process_screen

game = FlappyBird(graphics="fixed")
p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=True, display_screen=True)
# Note: if you want to see you agent act in real time, set force_fps to False. But don't use this setting for learning, just for display purposes.

p.init()
reward = 0.0
UP = 119
nb_games = 10
save = False
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

if save:
    folder = "data/"

    filehander = open(folder + "images.pickle", "wb")
    pickle.dump(images, filehander)
    filehander.close()

    for i, a in enumerate(actions):
        if a == UP:
            actions[i] = 1

    filehander = open(folder + "actions.pickle", "wb")
    pickle.dump(actions, filehander)
    filehander.close()

    filehander = open(folder + "rewards.pickle", "wb")
    pickle.dump(rewards, filehander)
    filehander.close()

    filehander = open(folder + "game.pickle", "wb")
    pickle.dump(game_id, filehander)
    filehander.close()
