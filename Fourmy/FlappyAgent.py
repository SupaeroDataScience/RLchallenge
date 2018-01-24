DIFFICULTY = 2

PREV = None
INDEX = 0
STRAIGHT = [None, None, None, None, 119, 119]


def FlappyPolicyDeter1(state, screen):
    py = state['player_y']
    npby = state['next_pipe_bottom_y']
    action = None
    if py >= npby-50:
        action = 119

    return action


def FlappyPolicyDeter2(state, screen):
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


def StraightAheadNoChaser(state, screen):
    # Corresponding velocity profile: [-8.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    global INDEX
    action = STRAIGHT[INDEX]
    INDEX += 1
    if INDEX >= len(STRAIGHT):
        INDEX = 0
    return action


POLICIES = [
    FlappyPolicyDeter1,
    StraightAheadNoChaser,
    FlappyPolicyDeter2,
]


def FlappyPolicy(state, screen):
    return POLICIES[DIFFICULTY](state, screen)
