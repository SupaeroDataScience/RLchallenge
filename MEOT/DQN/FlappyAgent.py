import numpy as np

import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.exposure import rescale_intensity

from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten
import graphviz

from collections import deque

list_actions = [0,119]
dqn = load_model('TrainG4_max.h5')
def process_screen(x):
    return (255 * resize(rgb2gray(x)[50:, :410], (84, 84))).astype("uint8")

def FlappyPolicy(state, screen):
    screen_x = process_screen(screen)
    stacked_x = deque([screen_x, screen_x, screen_x, screen_x], maxlen=4)
    x = np.stack(stacked_x, axis=-1)
    action = list_actions[np.argmax(dqn.predict(np.expand_dims(x,axis=0)))]
    return action


