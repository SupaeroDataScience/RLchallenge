import numpy as np
from keras.models import load_model
from collections import deque
from ple.games.flappybird import FlappyBird
from ple import PLE
from skimage import transform, color

bird_model = load_model('pixel_bird.dqf')
game = FlappyBird()
p = PLE(game, fps=30, frame_skip=1, num_steps=1)
list_actions = p.getActionSet()
size_img = (80,80)

frames = deque([np.zeros(size_img),np.zeros(size_img),np.zeros(size_img),np.zeros(size_img)], maxlen=4)

def process_screen(screen):
    return 255*transform.resize(color.rgb2gray(screen[60:, 25:310,:]),(80,80))

def FlappyPolicy(state, screen):
    global bird_model
    global frames
    global list_actions

    x = process_screen(screen)
    # Reset the frames deque if a new game is started
    if not np.any(x[10:,:]): # new game <=> black image in front of Flappy
        frames = deque([np.zeros(size_img),np.zeros(size_img),np.zeros(size_img),np.zeros(size_img)], maxlen=4)

    frames.append(x)
    frameStack = np.stack(frames, axis=-1)
    a = list_actions[np.argmax(bird_model.predict(np.expand_dims(frameStack,axis=0)))]
    return a