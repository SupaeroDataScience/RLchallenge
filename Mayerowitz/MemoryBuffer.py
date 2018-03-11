# Standard packages
import numpy as np
from collections import deque

# Import params
from params import *


"""
An experience replay buffer using numpy arrays
"""
class MemoryBuffer:

    def __init__(self, length, screen_shape, action_shape):
        """
        Init the arrays to use (frames, rewards, actions, etc.)

        length          -- the max size of the arrays to use
        screen_shape    -- the shape of the images in input (default : 80x80)
        action_shape    -- the shape of the different actions we can perform (default: 2)
        """
        # Get dimensions
        self.length = length
        self.screen_shape = screen_shape
        self.action_shape = action_shape
        # Starting (x) and resulting (y) states | Actions | Rewards
        self.screens_x  = np.zeros((length,) + screen_shape, dtype=np.uint8)
        self.screens_y  = np.zeros((length,) + screen_shape, dtype=np.uint8)
        self.actions    = np.zeros((length,) + action_shape, dtype=np.uint8)
        # We use negative rewards (if the flappy dies), so use a signed type (int8)
        self.rewards    = np.zeros((length,1), dtype=np.int8)
        # True if resulting state is terminal (game over), otherwise false
        self.terminals = np.zeros((length,1), dtype=np.bool)
        self.terminals[-1] = True
        # Points one position past the last inserted element
        self.index = 0
        # Current size of the buffer
        self.size = 0


    def append(self, s, a, r, s_, d):
        """
        Append new frames into the memory buffer

        s   -- the current screen
        a   -- the action to perform
        r   -- the reward
        s_  -- the next screen
        d   -- (done) a boolean telling if the game is over or not
        """
        self.screens_x[self.index] = s
        self.actions[self.index] = a
        self.rewards[self.index] = r
        self.screens_y[self.index] = s_
        self.terminals[self.index] = d
        self.index = (self.index + 1) % self.length
        self.size = np.min([self.size + 1, self.length])


    def stacked_frames(self, index, x_frames):
        """
        Get 4 stacked frames (x or y screens, depending on a parameter)

        index       -- the index of the array we want to have a look at
        x_frames    -- a boolean telling if we should consider the screens_x or screens_y variable
        """
        # Define a deque with a fixed-size of 4 elements (store 4 frames)
        frames = deque(maxlen=STACK_SIZE)
        pos = index % self.length
        for _ in range(STACK_SIZE):
            # Get x or y frames, depending on the parameter
            frame = self.screens_x[pos] if x_frames == True else self.screens_y[pos]
            frames.appendleft(frame)
            test_pos = (pos - 1) % self.length
            if self.terminals[test_pos] == False:
                pos = test_pos
        return np.stack(frames, axis=-1)


    def stacked_frames_x(self, index):
        """ Get 4 stacked frames from screens_x array """
        return self.stacked_frames(index=index, x_frames=True)


    def stacked_frames_y(self, index):
        """ Get 4 stacked frames for screens_y array"""
        return self.stacked_frames(index=index, x_frames=False)


    def minibatch(self, size):
        """
        Generate a minibatch and return it to the user

        size -- the size of the batch we want to create
        """
        idx = np.random.choice(self.size, size=size, replace=False)
        x = np.zeros((size,) + self.screen_shape + (STACK_SIZE,))
        y = np.zeros((size,) + self.screen_shape + (STACK_SIZE,))
        for i in range(size):
            x[i] = self.stacked_frames_x(index=idx[i])
            y[i] = self.stacked_frames_y(index=idx[i])
        return x, self.actions[idx], self.rewards[idx], y, self.terminals[idx]
