PREV_ACTION = None


def FlappyPolicy(state, screen):
    global PREV_ACTION
    action = PREV_ACTION
    PREV_ACTION = None

    y = state['player_y']
    if y >= state['next_pipe_bottom_y'] - 64 :
        PREV_ACTION = 119

    return action
