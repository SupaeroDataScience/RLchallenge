# Standard packages
import numpy as np
from enum import Enum

# Import constant
from constant import *

# Flappy Bird environment
from ple.games.flappybird import FlappyBird
from ple import PLE


# import the  class for the replay memory
from collections import deque 

# Import the library that allows to transforl the image
from skimage.color import rgb2gray
from skimage.transform import resize



# An enumeration to get the mode we want to play, here we are defining three different modes:
# training, playing or testing purposes
class Mode(Enum):
    TRAIN = 1
    TEST = 2
    PLAY = 3


# A class named that has all the functions required to operate a fixed-size stack
class Stack:

    # Initialize the stack
    def __init__(self, frame, length):
        self.length = length
        self.deque = self.init_deque(frame, length)
        self.stack = self.convert_to_stack(deque=self.deque)
        
    # Reset the stack with a new frame
    def reset(self, frame):
        self.stack = self.convert_to_stack(deque=self.deque)
        
    # Append a new element to the stack 
    def append(self, frame):
        self.deque.append(frame)
        self.stack = self.convert_to_stack(deque=self.deque)
        return self.stack

    # Initialize a fixed-size deque
    def init_deque(self, frame, length):
        return deque([frame for _ in range(length)], maxlen=length)

    # Reorganise the deque into a stack
    def convert_to_stack(self, deque):
        return np.stack(deque, axis=-1)



# This functiona allows us to convert RGB to greyscale mainly, and to resiize the frame
def transform_screen(frame):
    # Crop the frame
    frame = frame[60:, 25:310, :]
    # Convert it in grey scale
    frame = rgb2gray(frame)
    # Resize the frame to a squared one (like in Nature article) TODO : size as a constant
    frame = resize(frame, IMG_SHAPE)
    # Then re-map the pixels from 0 to 255 (instead of 0 to 1)
    frame = 255 * frame
    # Return the processed frame
    return frame

# Initation of the game and environment
def init_flappy_bird(mode, graphics="fixed"):

    # use "Fancy" for full background, random bird color and random pipe color,
    # use "Fixed" (default) for black background and constant bird and pipe colors.
    game = FlappyBird(graphics=graphics)

    # Set parameters, depending on the mode specified
    force_fps = (mode == Mode.TRAIN)
    display_screen = (mode == Mode.PLAY)

    # Note: if you want to see you agent act in real time, set force_fps to False.
    # But don't use this setting for learning, just for display purposes.
    env = PLE(game,
              fps=30,
              frame_skip=1,
              num_steps=1,
              force_fps=force_fps,
              display_screen=display_screen)

    # Init the environment (settings, display...) and reset the game
    env.init()
    env.reset_game()

    return game, env
