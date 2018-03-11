# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 15:02:59 2018

@author: Julio Hernanz González
"""

#### Creation of a Convolutional Neural Network Model for the Flappy Bird PLE game.
#### This is the training file. The bird plays the game a NB_GAMES and 
#### saves the information, updating the created model.

### I HAVE NOT ARRIVED TO A SUCCESFULL NEURAL NETWORK. IT DOESN'T CONVERGE. 

#%% IMPORTS

# We import the game
from ple.games.flappybird import FlappyBird
from ple import PLE

import random
import numpy as np

from collections import deque

# To work with the screen image
import skimage as skimage
from skimage import color, transform, exposure

# Keras allows us to create the neutal network
from keras import initializers
from keras.initializers import normal, identity
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
import tensorflow as tf

#%% NEURAL NETWORK

# This function allows us to create the model. The network is made of several
# layers. Conv2D are convolutional layers that use a filter to create an activation
# map (or feature map). We also use Rectified Linear Units ReLU and pooling layers.

def buildmodel():
    print("Now we build the model")
    model = Sequential()
    model.add(Conv2D(32, (8, 8), strides=(4, 4), padding='same',input_shape=(img_rows,img_cols,img_channels)))  #80*80*4
    model.add(Activation('relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(2))
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse',optimizer=adam)
    print("We finish building the model")
    return model


#%% CODE CORE


# PARAMETERS
ACTIONS = [0, 119] # valid actions (don't flap, flap)
GAMMA = 0.99 # decay rate of past observations
LEARNING_RATE = 0.0001 # alpha
OBSERVATION = 3200. # timesteps to observe before training
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
NB_EPISODES = 10000 # Number of episodes

img_rows, img_cols = 80, 80
img_channels = 4 # We stack 4 frames

# We start the backend
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
from keras import backend as K
K.set_session(sess)

# We build the nn model
model = buildmodel()

# We initialise the game
game = FlappyBird(graphics="fixed")
p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=True, display_screen=True)
p.init()
episode_counter = 0 
counter = 0 # counter to control the reduction of epsilon

# store the previous observations in replay memory
D = deque()

# First action, don't flap.
p.act(ACTIONS[0])
x_t = p.getScreenRGB()
terminal = p.game_over()

x_t = skimage.color.rgb2gray(x_t)
x_t = skimage.transform.resize(x_t,(80,80))
x_t = skimage.exposure.rescale_intensity(x_t,out_range=(0,255))

s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

#In Keras, need to reshape
s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  #1*80*80*4

#We go to training mode
epsilon = INITIAL_EPSILON
t = 0

loss = 0
Q_sa = 0
action_index = 0
r_t = 0

# We will be playing the game for a fixed number of episodes.
while (episode_counter <= NB_EPISODES):
    
    # We reset the game in case there has been a game over.
    if p.game_over():
        p.reset_game()
        
    # Choose an action epsilon greedy
    if random.random() <= epsilon:
        # Random action
        action_index = random.randrange(2)
        a_t = ACTIONS[action_index]
    else:
        q = model.predict(s_t)   # s_t is a stack of 4 images, we get the prediction
        max_Q = np.argmax(q)
        action_index = max_Q
        a_t = ACTIONS[action_index]
    counter += 1

    # Epsilon reduction (only if observation is done)
    if counter % 500 == 0 and t > OBSERVATION:
        epsilon *= 0.9

    # Run action and get next state and reward
    r_t = p.act(a_t)
    terminal = p.game_over()
    x_t1_colored = p.getScreenRGB()

    x_t1 = skimage.color.rgb2gray(x_t1_colored)
    x_t1 = skimage.transform.resize(x_t1,(80,80))
    x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))

    x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1) #1x80x80x1
    s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

    # We store the information in D
    D.append((s_t, action_index, r_t, s_t1, terminal))
    if len(D) > REPLAY_MEMORY:
        D.popleft()

    # Training when t is bigger than OBSERVATION (we have already observed enough)
    if t > OBSERVATION:
        
        # Pick the minibatch for the training (size is BATCH)
        minibatch = random.sample(D, BATCH)

        inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))   #32, 80, 80, 4
        targets = np.zeros((inputs.shape[0], len(ACTIONS)))  #32, 2

        # Experience replay
        for i in range(0, len(minibatch)):
            
            state_t = minibatch[i][0]
            action_t = minibatch[i][1] # Action index
            reward_t = minibatch[i][2]
            state_t1 = minibatch[i][3]
            terminal = minibatch[i][4]

            inputs[i:i + 1] = state_t    # We save s_t

            targets[i] = model.predict(state_t)  # Prediction
            Q_sa = model.predict(state_t1)

            if terminal:
                targets[i, action_t] = reward_t # if terminated, only equals reward
            else:
                targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)
                
        loss += model.train_on_batch(inputs, targets)
        
        # End of the episode
        episode_counter += 1
        
        # Control print
        if episode_counter % 100 == 0:
            print("Episode number:",episode_counter)

    # New state and time step + 1
    s_t = s_t1
    t = t + 1

    # We save the progress every 1000 iterations
    if t % 1000 == 0:
        print("Now we save the model") #☺ Control print
        model.save("model.h5", overwrite=True)
    