#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 13:58:56 2018

@author: paul
"""

from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np
import matplotlib.pyplot as plt
import pygame
from time import time
import pickle
from deepqn import process_screen, MemoryBuffer, clip_reward


def gamer_policy():
    timeout = time() + 0.1
    while time() < timeout:
        events = pygame.event.get()
        for event in events :
            if event.type == pygame.KEYUP:
                return UP
    return NOPE


def game_and_save(nb_frames, name, p):

    my_memory = MemoryBuffer(nb_frames)

    while len(my_memory.actions) < nb_frames:
        p.reset_game()
      
        while not p.game_over():
            screen = p.getScreenRGB()
            x = process_screen(screen)
            action = gamer_policy()
            reward = clip_reward(p.act(action))
            my_memory.append(x, action, reward)

            if len(my_memory.actions) > nb_frames:
                break
  
    filehandler = open(name + '.pickle', 'wb')
    pickle.dump(my_memory, filehandler)
    filehandler.close()


"""""""""""""""""""""""""""""""""""""""""""""""""""MAIN"""""""""""""""""""""""""""""""""""""""

if __name__ == "__main__":

    game = FlappyBird(graphics="fixed")
    p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=True, display_screen=True)

    p.init()
    ACTIONS = p.getActionSet()
    UP = ACTIONS[0]
    NOPE = ACTIONS[1]
    screen = p.getScreenRGB()
    x = process_screen(screen)
    plt.imshow(x, cmap="gray")
    # game_and_save(1000, "my_played_buffer", p)


