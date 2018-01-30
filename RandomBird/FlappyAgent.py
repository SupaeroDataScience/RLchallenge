import numpy as np

def FlappyPolicy(state, screen):
    action=None
    if(np.random.randint(0,2)<1):
        action=119
    return action


