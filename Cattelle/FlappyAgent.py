import numpy as np
from keras import models

from config import Config as config
from utils import StateHolder

# Initialise dqn
dqn = models.load_model(config.MODEL_FILENAME)

stateholder = StateHolder()


def FlappyPolicy(_, screen):
    """
    Main game policy, define the behaviour of the agent
    Args:
        _ (dict) : The state vector of the simulator, ignored here
        screen (numpy.ndarray): Current state of the screen (RGB matrix)

    Returns:
        action (int): The action to take
    """
    stateholder.append(screen)
    state = stateholder.get_dqn_input()

    Q = dqn.predict(state)  # Expect a (no_samples, history_length, 84, 84) input

    return np.argmax(Q) * 119  # argmax is either 0 or 1 with convention 0: no-op; 1: flap
