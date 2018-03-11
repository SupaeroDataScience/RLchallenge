#imports
from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, Dropout
from keras import initializers, optimizers
from PIL import Image
np.set_printoptions(precision=3)
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=3)
from skimage.color import rgb2gray
from skimage.transform import resize
from collections import deque

#load dqn
dqn = load_model('network-600000.h5')

#initialize stack
stacked_x = deque([], maxlen=4)
    
#functions
def greedy_action(network, x):
    Q = network.predict(np.array([x]))
    return np.argmax(Q)

def process_screen(x):
    h = 130
    w = 80
    return 256*resize(rgb2gray(x), (w,h))[int(w/6):,0:int(4/5*h)]

def int_to_action(n):
    if n == 0:
        return(None)
    if n == 1:
        return(119)

def FlappyPolicy(state, screen):
    screen_x = process_screen(screen) #processes screen
    if len(stacked_x) == 0: #if stack is empty, stacks 3 initials screen
        stacked_x.append(screen_x)
        stacked_x.append(screen_x)
        stacked_x.append(screen_x)
    stacked_x.append(screen_x) #stack last screen
    x = np.stack(stacked_x, axis=-1)
    action = int_to_action(greedy_action(dqn, x)) #call dqn for action
    return action


