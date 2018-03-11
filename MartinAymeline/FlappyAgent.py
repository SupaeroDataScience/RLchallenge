import numpy as np
from keras.models import load_model
from collections import deque
from utilities import process_screen, greedy_action


stacked = []
calls = 0
DQN = load_model('model_dqn_new_65000.h5')
possible_actions = [119,None]


def FlappyPolicy(state, screen):
    global stacked
    global calls
    global DQN
    global action
    
    calls = calls + 1
    processed_screen = process_screen(screen)
    
    if (calls == 1) :
        # stack the 4 last frames
        stacked = deque([processed_screen,processed_screen, \
                         processed_screen,processed_screen], maxlen=4)
        x = np.stack(stacked, axis=-1)
        
    else :
        stacked.append(processed_screen)
        x = np.stack(stacked, axis=-1)
    
    Q = DQN.predict(np.array([x]))
    
    return possible_actions[np.argmax(Q)]


