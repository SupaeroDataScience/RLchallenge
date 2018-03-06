#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 13:58:56 2018

@author: paul
"""

from ple.games.flappybird import FlappyBird
from ple import PLE
import pygame
from time import time
import pickle
from deepqn import process_screen, MemoryBuffer, clip_reward, memory_size


def gamer_policy():
    timeout = time() + 1/24
    while time() < timeout:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYUP:
                return 0
    return 1


def game_and_save(nb_frames, name, p):

    my_memory = MemoryBuffer(memory_size)
    i=0
    while len(my_memory.actions) < nb_frames:
        p.reset_game()
        p.act(0)
        while not p.game_over():
            print(i)
            screen = p.getScreenRGB()
            screen = process_screen(screen)
            idx = gamer_policy()
            action = ACTIONS[idx]
            reward = clip_reward(p.act(action))
            my_memory.append(screen, idx, reward)
            i += 1
            if len(my_memory.actions) > nb_frames:
                break

    filehandler = open(name + '.pickle', 'wb')
    pickle.dump(my_memory, filehandler)
    filehandler.close()


"""""""""""""""""""""""""""""""""""""""""""""""""""MAIN"""""""""""""""""""""""""""""""""""""""

if __name__ == "__main__":
    nf = 20000
    game = FlappyBird(graphics="fixed")
    p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=True, display_screen=True)

    p.init()
    ACTIONS = p.getActionSet()
    UP = ACTIONS[0]
    NOPE = ACTIONS[1]
    game_and_save(nf, "new_played_buffer_{}".format(nf), p)


