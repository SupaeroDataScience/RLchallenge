import numpy as np
from keras.models import load_model
from challenge_utils import process_screen
from collections import deque

deepQnet = load_model('model.h5')
list_actions = [0, 119]
size_img = (72,101)
frameDeque = deque([np.zeros(size_img),np.zeros(size_img),np.zeros(size_img),np.zeros(size_img)], maxlen=4)

def FlappyPolicy(state, screen):
    global deepQnet
    global frameDeque
    x = process_screen(screen)
    #frameDeque.append(np.ones(size_img))
    #print(frameDeque) 
    frameDeque.append(x)
    frameStack = np.stack(frameDeque, axis=-1)
    #frameStack = np.stack([x]*4,axis=-1) # STUPID : declare a glob variable and keep previous screens
    a = list_actions[np.argmax(deepQnet.predict(np.expand_dims(frameStack,axis=0)))]

    return a # Should return an action
