import numpy as np

Qlearning = dict()
Qsarsa = dict()

def FlappyPolicy(state, screen):
    """
    Returns an action for each timestep depending on the game state:
        'None' for doing nothing;
        '119' for jumping
    """
    action = actTDLambda(state)

    return( 119 if action else 0 )



def actQlearning(state):
    
    global Qlearning

    if not bool(Qlearning):
        Qlearning = np.load('Qlearning.npy')
       
    s1, s2, s3 = toDiscreteRef(state)
    key = str(s1)+'|'+str(s2)+'|'+str(s3)
    
    if Qlearning[()].get(key) == None:
        return 0

    return Qlearning[()][key][0] < Qlearning[()][key][1]


def actTDLambda(state):

    global Qsarsa

    if not bool(Qsarsa):
        Qsarsa = np.load('Qsarsa.npy')

    s1, s2, s3 = toDiscreteRef(state)
    key = str(s1)+'|'+str(s2)+'|'+str(s3)
    
    if Qsarsa[()].get(key) == None:
        return 0

    return Qsarsa[()][key][0] < Qsarsa[()][key][1]


def toDiscreteRef(state):
    """
    Converts the game state variables into the custom discrete variable state for the
    Q-learning approach.
    """

    s1 = state['next_pipe_bottom_y'] - state['player_y']
    s2 = state['next_pipe_dist_to_player']
    s3 = state['player_vel']
    
    return int(s1-s1%10), int(s2-s2%20), int(s3-s3%2)
