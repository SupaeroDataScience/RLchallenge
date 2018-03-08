POLICY = 4

PREV = None
INDEX = 1
STRAIGHT = [None, None, None, None, 119, 119]
# STRAIGHT = [None, None, None, None, 119, None, 119]


def policy_deter1(state, screen):
    py = state['player_y']
    npby = state['next_pipe_bottom_y']
    action = None
    if py >= npby-50:
        action = 119

    return action


def policy_deter2(state, screen):
    global PREV
    next_action = PREV
    py = state['player_y']
    npby = state['next_pipe_bottom_y']
    if py >= npby-60:
        PREV = 119
    else:
        PREV = None
    print(next_action)

    return next_action


def straight_ahead_no_chaser(state, screen):
    # Corresponding velocity profile: [1.0, 2.0, 3.0, 4.0, -8.0, 0.0]
    global INDEX
    py = state['player_y']
    print(py)
    action = STRAIGHT[INDEX]
    INDEX += 1
    if INDEX >= len(STRAIGHT):
        INDEX = 0
    return action


def always_up(state, screen):
    return 119


def always_down(state, screen):
    print(state)
    return None


POLICIES = [
    policy_deter1,
    policy_deter2,
    straight_ahead_no_chaser,
    always_up,
    always_down,
]


def FlappyPolicy(state, screen):
    return POLICIES[POLICY](state, screen)
