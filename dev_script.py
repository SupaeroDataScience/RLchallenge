#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 13:58:56 2018

@author: paul
"""

from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np

game = FlappyBird()
p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=False, display_screen=True)

p.init()


