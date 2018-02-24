""" Dumb policy so you can get familiar with the game """

PREV_ACTION = None


def FlappyPolicy(state, screen):
    """ Policy based on being in the middle of the pipe.

    :param state
    :param screen (not used)
    :return action (no jump (0) or jump (119))
    """
    global PREV_ACTION
    action = PREV_ACTION
    PREV_ACTION = None

    y = state['player_y']
    if y >= state['next_pipe_bottom_y'] - 64:
        PREV_ACTION = 119

    return action
