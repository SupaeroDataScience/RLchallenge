from collections import deque

import numpy as np
from PIL import Image

from config import Config as config


def process_screen(screen):
    """
        Process the screen state to a simpler version that can be fed to the DQN predictor

        The processing is as follows:
            1. Convert to grayscale
            2. Crop to 405x288
            3. Downscale and rescale to 84x84
            4. Normalise pixel values from [0,255] to [0,1]
    Args:
        screen (np.ndarray): A RGB matrix

    Returns:
        im (np.ndarray): Processed screen
    """

    # Indexing convention varies between PIL and numpy
    screen = np.swapaxes(screen, 0, 1)
    # Load the array in PIL
    im = Image.fromarray(screen, 'RGB')
    # Convert to grayscale
    im = im.convert(mode='L')
    # Crop
    im = im.crop((0, 0, 405, 288))
    # Downscale and resize
    im = im.resize((84, 84))
    # Normalise
    im = np.array(im) / 255

    return im


class StateHolder:
    """
        A simple class designed to keep track of the previous frames in order to build a valid input for the DQN (
        input shape of (no_sample, history_length, 84, 84)
    """
    state = deque(maxlen=config.HISTORY_LENGTH)

    def append(self, screen):
        """
            Append a new frame to the holder. Handles the initial insertion case gracefully.
        Args:
            screen (np.ndarray): The current frame
        """
        if len(self.state) == 0:
            # Initial insertion
            # No need to handle terminal cases as we don't restart from a game over, we just start a whole new game
            self.state = deque([process_screen(screen)] * 4, maxlen=config.HISTORY_LENGTH)

        else:
            self.state.append(process_screen(screen))

    def get_dqn_input(self):
        """
            Return a numpy array ready to be fed to the DQN
        Returns:
            input_arr (np.ndarray): Input array for the DQN
        """
        state_arr = np.array([elt for elt in self.state])

        return state_arr.reshape((1, config.HISTORY_LENGTH, 84, 84))
