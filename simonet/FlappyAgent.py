from state import new_state
import numpy as np
import pickle

opened = 0
Q = np.zeros((18,30,21,2))

def FlappyPolicy(state, screen):

    global opened
    global Q

    if not opened :
        file = open("Qtrained",'rb')
        Q = pickle.load(file)
        opened = 1

    s = new_state(state)
    action = np.argmax(Q[s[0],s[1],s[2]][:])
    return action*119


