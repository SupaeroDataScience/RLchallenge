# Some imports
import numpy as np
from keras.models import load_model
from params import *
from utils import process_screen, Stack

# Load the model
dqn = load_model('./Models/dqn.h5')

# Empty image
img = np.zeros((IMG_SHAPE))

# Init the stack
x = Stack(frame=img, length=STACK_SIZE)

def FlappyPolicy(state, screen):

    # Process screen
    S = process_screen(frame=screen)

    # Reinitialize the deque if we start a new game
    if not np.any(S):
        x.reset(frame=img)

    # Append the frame to the stack
    x.append(frame=S)

    # Predict an action to perform
    A = np.argmax(dqn.predict(np.expand_dims(x.stack, axis=0)))

    # Return action we want to perform
    return ACTIONS[A]
