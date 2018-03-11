import numpy as np
from collections import deque
from skimage import color, transform
from keras.models import load_model

stackedX = []
call = 0
actions = [119, None]
dqn = load_model('dqn-925k.h5')
# Choose a new action every REPEAT call
REPEAT = 2
lastAction = None

def processScreen(screen):
    """ Resize and gray-ify screen """
    return 255*transform.resize(color.rgb2gray(screen[60:,25:310,:]),(80,80))
    
def FlappyPolicy(state, screen):
    global stackedX, call, actions, dqn, lastAction
    
    screenX = processScreen(screen)

    if call == 0: 
        stackedX = deque([screenX]*4, maxlen=4)
        x = np.stack(stackedX, axis=-1)
    else:
        stackedX.append(screenX)
        x = np.stack(stackedX, axis=-1)
        
    Q = dqn.predict(np.array([x]))
    
    if call % REPEAT == 0 or REPEAT == 1:
        lastAction = actions[np.argmax(Q)]
    call += 1
    return lastAction
