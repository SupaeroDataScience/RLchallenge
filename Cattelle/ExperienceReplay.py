from collections import deque

import numpy as np

from utils import process_screen


class ExperienceReplay:
    """
        This class defines a handy structure to store and handle the experience replay memory

        It provides the following actions:
            * Process the screen (convert to grayscale, downscale, crop)
            * Update the underlying experience replay array
            * Sample randomly the array to yield a minibatch
            * Append new state to the array
    """

    def __init__(self, size, history_length=4, minibatch_size=32):
        """
            Initialise the underlying array holding the experience replay memory

            One sample is defined as a (s,a,r,s',d) tuple where one state (s) corresponds to the history_length last
            frames stacked together. Each state is implemented as deque which prevents having to handle the maximum
            length and speeds up access time to both ends of the queue.

            The memory is implemented as a deque as well, and is filled from left to right. The right-most sample is
            thus always the newest one.

        Args:
            size (int): Total number of state
            history_length (int): Number of frame to keep in one state (stacked together). Default is 4
            minibatch_size (int): Number of samples in a minibatch
        """
        self.memory = deque(maxlen=size)
        self.history_length = history_length
        self.size = size
        self.minibatch_size = minibatch_size

    def append_sample(self, s, a, r, s_new, d):
        """
            Append a new sample to the ER memory.

            The screen states will be processed and appended to the correct stacks
        Args:
            s (np.ndarray): Raw (unprocessed) screen state
            a (int): Action taken at state s leading to state s_new
            r (float): Reward for taking action a at state s
            s_new (np.ndarray): Raw (unprocessed) screen state
            d (bool): True if the game is in a terminal state (game over), False otherwise
        """

        if len(self.memory) == 0 or self.memory[-1][4] is True:
            # We handle the initial insertion or the first one after a terminal differently
            # The initial state in this case is 4 times the same frame
            s = process_screen(s)
            state = deque([s] * self.history_length, maxlen=self.history_length)  # state = [s, s, ..., s]

            state_new = state.copy()
            state_new.append(process_screen(s_new))  # state_new = [s, s, ..., s_new]

            self.memory.append((state, a, r, state_new, d))

        else:
            # Grab the last sample recorded
            last_sample = self.memory[-1]

            # Build the new state (stack)
            new_state = last_sample[3].copy()
            new_state.append(process_screen(s_new))

            # And append to the memory
            self.memory.append((last_sample[3], a, r, new_state, d))

    def minibatch(self):
        """
            Randomly samples a minibatch of size minibatch_size and returns it
        Returns:
            minibatch (np.ndarray): Randomly sampled minibatch
        """

        # Get the current size of the memory
        size = len(self.memory)

        if size < self.minibatch_size:
            raise IndexError(f'minibatch_size ({self.minibatch_size}) is larger than the current size of the ER '
                             f'memory ({size})')

        # er.memory is not 1D thus we cannot sample it directly, instead we sample indices and build back and array
        indices = np.random.choice(size, self.minibatch_size)

        return np.array([self.memory[i] for i in indices]).T
