import numpy as np
from keras.models import load_model
from collections import deque
from ple.games.flappybird import FlappyBird
from ple import PLE
from skimage.color import rgb2gray
from skimage.transform import resize

#Rq: il se peut que la version de keras 2.1.5 soit spécifiquement demandée pour ouvrir le DQN, l'entrainement s'est fait
#sur le cloud avec cette version de keras


DQN = load_model('flappy_brain.h5')


game = FlappyBird(graphics="fixed")
p = PLE(game, fps=30, frame_skip=1, num_steps=1)

list_actions = p.getActionSet()

DequeFX = deque([np.zeros((80,80)),np.zeros((80,80)),np.zeros((80,80)),np.zeros((80,80))], maxlen=4)

def process_screen(screen):
    return 255*resize(rgb2gray(screen[60:, 25:310,:]),(80,80))

def FlappyPolicy(state, screen):
    
    global DQN
    global DequeFX
    global list_actions

    x = process_screen(screen)
    
    # If new game, build new stacked frames 
    if not np.any(x[10:,:]): # to know if x is the initial flappy position
        DequeFX = deque([np.zeros((80,80)),np.zeros((80,80)),np.zeros((80,80)),np.zeros((80,80))], maxlen=4)
        
    #else:
    DequeFX.append(x)
    FramesFX = np.stack(DequeFX, axis=-1)
    act = list_actions[np.argmax(DQN.predict(np.expand_dims(FramesFX,axis=0)))]

    return act # Return the action to perform
