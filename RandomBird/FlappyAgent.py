'''
FLAPPY AGENT for Supaero RL Challenge
Author : Timon (ft. Thomas)

Imports current model from model folder.
Implements decision function using trained model.
Debug options available.
'''

# IMPORTS
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.models import load_model
from collections import deque

# DECLARATIONS

# Initialize global values
STACK = []
ITER = 0
PREDICT_NET = load_model('models/current_model.h5')
ACTIONS = [119, None]


# Utilitary functions
def process_screen(screen):
    '''
    Go from phone-format color image to square grayscale.
    MEMO : any update in this function must me duplicated
    in the training file
    '''
    screen = screen[50:270, :320]
    return 255*resize(rgb2gray(screen), (80,80))

def first_input(first_screen, length=4):
    '''
    Initialize deque of max length {length}
    and fills it with first screen
    '''
    deq = deque([first_screen for i in range(length)], maxlen=length)
    return deq

# Policy
def FlappyPolicy(state, screen, print_frequency = 0, print_debug=False):
    '''
    Returns selected action based upon screen.
    4-steps, screen-only version
    '''

    # Global inplace modification will happen
    global STACK
    global ITER
    global PREDICT_NET
    global ACTIONS

    # Start process
    ITER += 1
    screen = process_screen(screen)

    if ITER == 1:
        STACK = first_input(screen)
        features = np.stack(STACK, axis=-1)
    else:
        STACK.append(screen)
        features = np.stack(STACK, axis=-1)

    Q_array = PREDICT_NET.predict(np.array([features]))
    selected_action = ACTIONS[np.argmax(Q_array)]

    if print_frequency > 0 and ITER % print_frequency == 1:
        print('STEP', iter)
        

    if print_debug : print('ITER:',ITER,'Q result:', Q_array)

    return selected_action
