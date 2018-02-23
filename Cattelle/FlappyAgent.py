import numpy as np


def FlappyPolicy(state, screen, train=False):
    """
    Main game policy, define the behaviour of the agent
    Args:
        state (dict): Current state of the emulator
        screen (numpy.ndarray): Current state of the screen (RGB matrix)
        train (bool): set to True to initiate training, otherwise (default) load the precomputed policy

    Returns:
        action (int): The action to take
    """

    action = None

    if train:
        # Setup the DQN here
        if np.random.randint(0, 2) < 1:
            action = 119

    else:
        # Load the precomputed optimal policy and return the action
        if np.random.randint(0, 2) < 1:
            action = 119

    return action
