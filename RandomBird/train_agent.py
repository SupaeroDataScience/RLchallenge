'''
Training routine for Supaero RL Challenge
Author : Timon (ft. Thomas)

Creates and trains a network for Flappy
'''

# IMPORTS
import numpy as np
import time
from ple import PLE
from ple.games.flappybird import FlappyBird
from matplotlib import pyplot as plt
from collections import deque

from MemoryBufferClass import MemoryBuffer
from keras.optimizers import Adam, SGD, RMSprop
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten
from skimage.color import rgb2gray
from skimage.transform import resize
from FlappyAgent import process_screen, first_input 

# HYPERPARAMETERS
NB_EPOCHS = 300000
MEMORY_SIZE = 1000000
BATCH = 32
GAMMA = 0.99
OBSERVE = 10000
EXPLORE = 1000000
INITIAL_EPSILON = 0.
LEARNING_RATE = 1e-4
DOUBLE_FREQ = 2500
ACCELERATE_TRAINING = 5

## Metaparameters, kinda?
DISPLAY_SCREEN = False
IMG_HEIGHT = 80
IMG_WIDTH = 80
print_freq = 1000
log_freq = 500
network_output = 'models/model.h5'
log_output = 'logs/logfile.csv'
verbose = False

# CONTROLLER UTILITARIES
def epsilon(step):
    '''
        Thomas's concept of a function for epsilon is really cool.
        Concept to remember.
    '''
    
    global OBSERVE
    global EXPLORE
    global INITIAL_EPSILON
    global FINAL_EPSILON
    
    if step < OBSERVE:
        return 1
    elif step < EXPLORE:
        slope = (FINAL_EPSILON - INITIAL_EPSILON)/(EXPLORE - OBSERVE)
        intercept = INITIAL_EPSILON - OBSERVE*slope
        return  slope*step + intercept
    else:
        return FINAL_EPSILON

def greedy_action(predictor, x):
    Q = predictor.predict(np.array([x]))
    return np.argmax(Q)
    
def epsilon_greedy_action(predictor, x, step):
    if np.random.rand() < epsilon(step):
        a = np.random.randint(2)
    else:
        a = greedy_action(convnet, x)
    return a

# MODEL
def make_sequential(print_sum=False):
    '''
    Stack all add commands for keras sequential
    Config : DQN paper (tho's conf)
    '''
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(8,8), strides=4, activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH,4)))
    model.add(Conv2D(filters=32, kernel_size=(4,4), strides=2, activation='relu'))
    model.add(Flatten())
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=2, activation='linear'))
    # try functional?
    return model

# Initialize dqn and dqn_target
model = make_sequential()
opt = RMSprop(lr = LEARNING_RATE)
model.compile(loss = 'mean_squared_error', optimizer = opt)
model.save(network_output)
model_target = load_model(network_output) #duplicate

# Load environment
game = FlappyBird(graphics='fixed') # that was helluvah move
p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=False, display_screen = DISPLAY_SCREEN)
p.init()
p.reset_game()
actions = p.getActionSet()
screen_x = process_screen(p.getScreenRGB())
STACK = first_input(screen_x)
features = np.stack(STACK, axis=-1)

'''screen_x = process_screen()
stacked_x = deque([screen_x, screen_x, screen_x, screen_x], maxlen=4)
x = np.stack(stacked_x, axis=-1)'''

# Init memory
replay_memory = MemoryBuffer(MEMORY_SIZE, (IMG_HEIGHT, IMG_WIDTH), (1,)) # don't quite like this one but...

# Ready log
logFile = open(log_output, 'w')
logFile.write('Step,Episode,Loss,Mean_Reward,Time \n')
game, episode_reward, mean_reward = 0, 0, 0
start_time = time.time()

# Passes
epoch, loss = 0, float('Inf')
while epoch < NB_EPOCHS:
    
    # select action
    a = epsilon_greedy_action(model, features, epoch)
    # get reward
    r = p.act(actions[a])
    episode_reward += r
    screen_y = process_screen(p.getScreenRGB())
    d = p.game_over()
    replay_memory.append(screen_x, a, r, screen_y, d)
    # train
    if epoch > BATCH and epoch % ACCELERATE_TRAINING == 0 and epoch  > OBSERVE:
        X,A,R,Y,D = replay_memory.minibatch(BATCH)
        QY = model_target.predict(Y)
        QYmax = QY.max(1).reshape((BATCH,1))
        update = R + GAMMA * (1-D) * QYmax
        QX = model.predict(X)
        QX[np.arange(BATCH), A.ravel()] = update.ravel()
        loss = float(model.train_on_batch(x=X, y=QX))

    # transfert weights between networks
    if epoch > 1 and epoch % DOUBLE_FREQ == 0:
        model.save(network_output)
        model_target = load_model(network_output)

    # prepare next transition
    if d==True:
        # update mean reward over episodes
        try:
            mean_reward += (1.0/game)*(episode_reward - mean_reward)
        except:
            if verbose: print('An error occured during reward update. Keeping previous result.')
        # restart episode
        episode_reward = 0
        p.reset_game()
        game += 1
        screen_x = process_screen(p.getScreenRGB())
        STACK = deque([screen_x, screen_x, screen_x, screen_x], maxlen=4)
        features = np.stack(STACK, axis=-1)
    else:
        # keep going
        screen_x = screen_y
        STACK.append(screen_x)
        features = np.stack(STACK, axis=-1)
        
    #Printing time
    if epoch%print_freq ==0 :
        print('epoch', epoch, '/', NB_EPOCHS, 'loss=',loss, 'epsilon=', epsilon(epoch))
    if epoch%log_freq ==0 :
        duration = time.time() - start_time
        logFile.write('{},{},{},{},{}\n'.format(epoch, game, loss, mean_reward, duration))

    #Increment
    epoch += 1
    
#Save and quit
print('Saving model')
model.save(network_output)

print('-/- \nTERMINATED \n-/-')

