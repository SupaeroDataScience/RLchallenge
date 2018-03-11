import numpy as np
from keras.models import load_model
from collections import deque
from skimage import transform,color
from ple.games.flappybird import FlappyBird
from ple import PLE

def ProcessScreen(x):
    return 255*transform.resize(color.rgb2gray(x[60:, 25:310,:]),(80,80))


birb = load_model('screeny_birb.dqf')
game = FlappyBird()
p = PLE(game, fps=30, frame_skip=1, num_steps=1)
list_actions = p.getActionSet()
img_dims = (80,80)

frameDeque = deque([np.zeros(img_dims),np.zeros(img_dims),np.zeros(img_dims),np.zeros(img_dims)], maxlen=4)

def FlappyPolicy(state, screen):
    global birb
    global frameDeque
    global list_actions

    x = ProcessScreen(screen)
    # Reinitialize the deque if we start a new game
    if not np.any(x[10:,:]): # if everything in front of Flappy is black
        frameDeque = deque([np.zeros(img_dims),np.zeros(img_dims),np.zeros(img_dims),np.zeros(img_dims)], maxlen=4)

    frameDeque.append(x)
    frameStack = np.stack(frameDeque, axis=-1)
    a = list_actions[np.argmax(birb.predict(np.expand_dims(frameStack,axis=0)))]

    return a 