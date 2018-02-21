import numpy as np
from keras.models import load_model
from challenge_utils import process_screen
from collections import deque
from ple.games.flappybird import FlappyBird
from ple import PLE

deepQnet = load_model('model.h5')
game = FlappyBird(graphics="fixed")
p = PLE(game, fps=30, frame_skip=1, num_steps=1)
list_actions = p.getActionSet()
size_img = (80,80)

frameDeque = deque([np.zeros(size_img),np.zeros(size_img),np.zeros(size_img),np.zeros(size_img)], maxlen=4)

def FlappyPolicy(state, screen):
    global deepQnet
    global frameDeque
    global list_actions

    x = process_screen(screen)
    # Reinitialize the deque if we start a new game
    if not np.any(x[10:,:]): # if everything in front of Flappy is black
        frameDeque = deque([np.zeros(size_img),np.zeros(size_img),np.zeros(size_img),np.zeros(size_img)], maxlen=4)

    frameDeque.append(x)
    frameStack = np.stack(frameDeque, axis=-1)
    a = list_actions[np.argmax(deepQnet.predict(np.expand_dims(frameStack,axis=0)))] #10 times quicker than np.array([])

    return a # Return the action to perform
