from collections import deque
from keras.models import load_model
import numpy as np
import sys
import COUSIN.PARAMS as params
from COUSIN.Tools import process_screen, greedy_action

# keep some information about previous games + load only once the model
# nn = load_model("Save/model_dqn_flappy2.h5")


def FlappyPolicy(state, screen):
    """ Policy based on a Neural Network. Use states.

    :param state
    :param screen (not used)
    :return action (no jump (0) or jump (119))
    """
    global nn
    print(state.values())
    print(dir(state))
    sys.exit()

    # a game starts again (black screen)


    # Use the Deep Q Network
    a = greedy_action(nn, x)  # 0 or 1

    return params.LIST_ACTIONS[a]
