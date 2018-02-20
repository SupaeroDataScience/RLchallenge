import numpy as np
from keras.models import load_model
from challenge_utils import process_screen
from collections import deque

deepQnet = load_model('model.h5')
#list_actions = [119, None]
list_actions = [None, 119] # Weirdly, this is the inverse of the order it has learnt...But works better 
size_img = (80,80)
frameDeque = deque([np.zeros(size_img),np.zeros(size_img),np.zeros(size_img),np.zeros(size_img)], maxlen=4)

def FlappyPolicy(state, screen):
    global deepQnet
    global frameDeque
    global list_actions

    x = process_screen(screen)

    frameDeque.append(x)
    frameStack = np.stack(frameDeque, axis=-1)
    a = list_actions[np.argmax(deepQnet.predict(np.expand_dims(frameStack,axis=0)))] #10 times quicker than np.array([])

    return a # Return the action to perform
