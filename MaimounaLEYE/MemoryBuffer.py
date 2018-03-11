# Import Standard Libraries
import numpy as np

#Constant parameters importation
from constant import *

# A class for the replay memory
from collections import deque


#Experience Memory Buffer
class MemoryBuffer:

    
    
    #Initiation of the arrays 
    def __init__(self, length, screen_shape, action_shape):
        self.length = length
        self.screen_shape = screen_shape
        self.action_shape = action_shape
        # Starting (x) and resulting (y) states, Rewards and Actions
        self.screens_x  = np.zeros((length,) + screen_shape, dtype=np.uint8)
        self.screens_y  = np.zeros((length,) + screen_shape, dtype=np.uint8)
        self.actions    = np.zeros((length,) + action_shape, dtype=np.uint8)
        self.rewards    = np.zeros((length,1), dtype=np.int8)
        self.terminals =  np.zeros((length,1), dtype=np.bool)
        self.terminals[-1] = True
        # Points one position past the last inserted element
        self.index = 0
        # Current size of the buffer
        self.size = 0



#Get stacked frames depending on the axis 
    def stacked_frames(self, index, x_frames):
        # Define a deque with a capacity of 4 frames
        frames = deque(maxlen=NB_FRAMES)
        pos = index % self.length
        for _ in range(NB_FRAMES):
            # Get x or y frames, depending on the parameter chosen
            frame = self.screens_x[pos] if x_frames == True else self.screens_y[pos]
            frames.appendleft(frame)
            test_pos = (pos - 1) % self.length
            if self.terminals[test_pos] == False:
                pos = test_pos
        return np.stack(frames, axis=-1)
    
    
    def stacked_frames_y(self, index):
        return self.stacked_frames(index=index, x_frames=False)

    def stacked_frames_x(self, index):
        return self.stacked_frames(index=index, x_frames=True)



   #Generate a new batch whose size is chosen by the user
    def minibatch(self, size):
        idx = np.random.choice(self.size, size=size, replace=False)
        x = np.zeros((size,) + self.screen_shape + (NB_FRAMES,))
        y = np.zeros((size,) + self.screen_shape + (NB_FRAMES,))
        for i in range(size):
            x[i] = self.stacked_frames_x(index=idx[i])
            y[i] = self.stacked_frames_y(index=idx[i])
        return x, self.actions[idx], self.rewards[idx], y, self.terminals[idx]

    #Append new frames in our memory buffer to update it 
    def append(self, s, a, r, s_, d):
        # Append current state,new state, action and rewards
        self.screens_x[self.index] = s
        self.actions[self.index] = a
        self.rewards[self.index] = r
        self.screens_y[self.index] = s_
        self.terminals[self.index] = d
        self.index = (self.index + 1) % self.length
        self.size = np.min([self.size + 1, self.length])

   
