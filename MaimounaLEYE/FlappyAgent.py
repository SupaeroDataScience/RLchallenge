#Import our constant  parameters and utilities
from constant import *
from utils import *

#Import the standard libraries
import numpy as np
from keras.models import load_model


#the first step is to load the trained model 
dqn = load_model('./model.h5')

# Empty image
img = np.zeros((IMG_SHAPE))

# Init the stack
x = Stack(frame=img, length=NB_FRAMES)

def FlappyPolicy(state, screen):

    # Process screen
    S = transform_screen(frame=screen)

    # Reinitialize the deque if we start a new game
    if not np.any(S):
        x.reset(frame=img)

    # Append the frame to the stack
    x.append(frame=S)

    # Predict an action to perform
    A = np.argmax(dqn.predict(np.expand_dims(x.stack, axis=0)))

    # Return action we want to perform
    return ACTIONS[A]
