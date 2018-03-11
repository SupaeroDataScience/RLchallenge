import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize

def FlappyPolicy(state, X, Q):

    actions = [119,None]
    
    # Greedy policy
    X = np.expand_dims(np.stack(X, axis=-1), axis=0)
    qa = Q.predict(X)
    a = np.argmax(qa)

    # Return greedy action
    return actions[a]


def process_screen(screen):
    return 255*resize(rgb2gray(screen[60:, 25:310, :]), (80, 80))
