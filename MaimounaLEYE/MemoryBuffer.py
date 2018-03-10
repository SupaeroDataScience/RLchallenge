# Standard packages
import numpy as np
from params import *

# A class for the replay memory
from collections import deque


"""
An experience replay buffer using numpy arrays
"""
class MemoryBuffer:

    """
    Init the arrays to use (frames, rewards, actions, etc.)
    """
    def __init__(self, length, screen_shape, action_shape):
        # Get dimensions
        self.length = length
        self.screen_shape = screen_shape
        self.action_shape = action_shape
        # Starting (x) and resulting (y) states | Actions | Rewards
        self.screens_x  = np.zeros((length,) + screen_shape, dtype=np.uint8)
        self.screens_y  = np.zeros((length,) + screen_shape, dtype=np.uint8)
        self.actions    = np.zeros((length,) + action_shape, dtype=np.uint8)
        self.rewards    = np.zeros((length,1), dtype=np.int8)
        # True if resulting state is terminal (game over), otherwise false
        self.terminals = np.zeros((length,1), dtype=np.bool)
        self.terminals[-1] = True
        # Points one position past the last inserted element
        self.index = 0
        # Current size of the buffer
        self.size = 0

    """
    Append new frames into the memory buffer
    """
    def append(self, s, a, r, s_, d):
        # Append current state, action, reward and new state into arrays
        self.screens_x[self.index] = s
        self.actions[self.index] = a
        self.rewards[self.index] = r
        self.screens_y[self.index] = s_
        self.terminals[self.index] = d
        self.index = (self.index + 1) % self.length
        self.size = np.min([self.size + 1, self.length])

    """
    Get 4 stacked frames on the axis defined in parameter
    """
    def stacked_frames(self, index, x_frames):
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
        return self.stacked_frames(index=index, x_frames=True)

    def stacked_frames_y(self, index):
        return self.stacked_frames(index=index, x_frames=False)

    """
    Generate a minibatch (size defined by the user)
    """
    def minibatch(self, size):
        idx = np.random.choice(self.size, size=size, replace=False)
        x = np.zeros((size,) + self.screen_shape + (STACK_SIZE,))
        y = np.zeros((size,) + self.screen_shape + (STACK_SIZE,))
        for i in range(size):
            x[i] = self.stacked_frames_x(index=idx[i])
            y[i] = self.stacked_frames_y(index=idx[i])
        return x, self.actions[idx], self.rewards[idx], y, self.terminals[idx]
