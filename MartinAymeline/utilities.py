import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, Activation, MaxPooling2D
from keras.optimizers import Adam
from keras.models import load_model
from collections import deque
from skimage import transform, color
from constantes import constantes as cst
import os, pickle


def process_screen(rgb_screen):
    # Initially, screen dimensions are (288, 512) #env.getScreenDims() and each
    # pixel of the image is a vector of its 3 color components.
    # Pipe is defined such that "pipe_gap = 100" and the gap could start randomly
    # between "pipe_min = int(pipe_gap/4) = 25" and "pipe_max = 
    # int(512 * 0.79 * 0.6 - pipe_gap / 2) = 193". Thus, for a given screen "Screen" 
    # whose dimensions are (288, 512), the real playing area is "Screen[:, 25:293]"
    # whose dimensions are (288, 268). We take some margins on both sizes to 
    # obtain a (240, 320) image.
    
    # PROCESSING : We convert the image to grayscale with a 256 color palette.
    # We crop it by keeping the useful playing area and then we downsample
    # the screen to a (80, 80) image.
    
    return 256*transform.resize(color.rgb2gray(rgb_screen)[50:270,0:320], (80,80))

def create_network():
    # Creation of the Convolutional Neural Network that will predict the Q-values
    dqn = Sequential()
    # The input is compound of the 4 last frames of the image whose size is (80, 80) 
    # 1st layer : convolutional layer with 80x80x4 input
    dqn.add(Conv2D(filters=16, kernel_size=(8,8), strides=4, padding='same', \
                   input_shape=(80,80,4), kernel_initializer='random_normal'))
    dqn.add(Activation("relu"))
    # 2nd layer : convolutional layer with ReLU activation
    dqn.add(Conv2D(filters=32, kernel_size=(4,4), strides=2, padding='same'))
    dqn.add(BatchNormalization())
    dqn.add(Activation("relu"))
    dqn.add(Flatten())
    # 3rd layer : fully connected layer with 256 ReLU units
    dqn.add(Dense(units=256))
    dqn.add(Activation("relu"))
    # Output layer : fully connected layer with 4 RelU units
    dqn.add(Dense(units=2, activation="linear"))
     
    # Network compilation 
    adam = Adam(lr=cst.alpha, beta_1=cst.beta_1, beta_2=cst.beta_2)
    dqn.compile(loss='mean_squared_error',optimizer=adam)
    
    # Network storing
    print(dqn.summary())
    dqn.save('model_dqn_new.h5')  
    
    return dqn 

    
def epsilon(step):
    if step < cst.observation:
        return 1
    elif step < 1e6:
        return cst.initial_eps - 9.9e-8*step 
    else:
        return cst.final_eps
    

def clip_reward(r):
    rr = 0
    if r>0:
        rr = 5 # When Flappy passa pipe
    if r<0:
        rr = -5 # When Flappy dies
    return rr

def greedy_action(network, x):
    Q = network.predict(np.array([x]))
    print("predict Q values :",Q)
    return np.argmax(Q)

def MCeval(env, trials, network, gamma):
    # Evaluate the performances of the network during the game
    possible_actions = env.getActionSet() # return [119, None]
    scores = np.zeros((trials))
    
    for i in range(trials):
        env.reset_game()
        
        screen_x = process_screen(env.getScreenRGB()) 
        stacked_x = deque([screen_x, screen_x, screen_x,screen_x], maxlen=4)
        x = np.stack(stacked_x, axis = -1)
        
        while not env.game_over():
            action = possible_actions[greedy_action(network, x)] 
            reward = env.act(action)
            screen_y = process_screen(env.getScreenRGB())
            scores[i] = scores[i] + reward
            
            if not env.game_over():
                # keep going
                screen_x = screen_y 
                stacked_x.append(screen_x)
                x = np.stack(stacked_x, axis = -1)   

    return np.mean(scores), np.max(scores)

