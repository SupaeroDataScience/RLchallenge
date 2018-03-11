from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop, sgd, Adam
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Conv2D, Flatten
import numpy as np
from collections import deque
import random
from IPython.display import clear_output
import matplotlib.pyplot as plt
np.set_printoptions(precision=3)
from skimage import transform, color

from ple.games.flappybird import FlappyBird
from ple import PLE


total_steps = 200000
replay_memory_size = 350000
mini_batch_size = 32
gamma = 0.99
evaluation_period = 10000
nb_epochs = total_steps // evaluation_period
epoch=-1

resume_training = False

## Deep Q-Network
if not resume_training:
    dqn = Sequential()
    #1st layer
    dqn.add(Conv2D(filters=16, kernel_size=(8,8), strides=4, activation="relu", input_shape=(80,80,4)))
    #2nd layer
    dqn.add(Conv2D(filters=32, kernel_size=(4,4), strides=2, activation="relu"))
    dqn.add(Flatten())
    #3rd layer
    dqn.add(Dense(units=256, activation="relu"))
    #output layer
    dqn.add(Dense(units=2, activation="linear"))
    dqn.compile(optimizer=Adam(lr=1e-4), loss='mean_squared_error')
else :
    dqn = load_model('pixel_bird.dqf')  #if training is already done

def epsilon(step):
    if step < 5e3:
        return 1
        # Then decrease linearly from 0.1 to 0.001 between
        # 10k and 1e6 steps
    elif step < 1e6:
        return (0.1 - 15e3*(1e-3-0.1)/(1e6-5e3)) + step * (1e-3-0.1)/(1e6-5e3)
    else:
        return 1e-3

def clip_reward(r):
    rr=0.05 # per-frame reward 
    if r>0:
        rr=1
    if r<0:
        rr=0
    return rr
    
def process_screen(screen):
    return 255*transform.resize(color.rgb2gray(screen[60:, 25:310,:]),(80,80))

def greedy_action(network, x):
    Q = network.predict(np.array([x]))
    return np.argmax(Q)

def MCeval(network, trials, length, gamma, p):
    scores = np.zeros((trials))
    for i in range(trials):
        #clear_output(wait=True)
        print("Trial",i,"out of",trials)
        p.reset_game()
        screen_x = process_screen(p.getScreenRGB())
        stacked_x = deque([screen_x, screen_x, screen_x, screen_x], maxlen=4)
        x = np.stack(stacked_x, axis=-1)
        for t in range(length):
            a = greedy_action(network, x)
            r = p.act(p.getActionSet()[a])
            r = clip_reward(r)
            screen_y = process_screen(p.getScreenRGB())
            scores[i] = scores[i] + gamma**t * r
            if p.game_over():
                # restart episode
                p.reset_game()
                screen_x = process_screen(p.getScreenRGB())
                stacked_x = deque([screen_x, screen_x, screen_x, screen_x], maxlen=4)
                x = np.stack(stacked_x, axis=-1)
            else:
                # keep going
                screen_x = screen_y
                stacked_x.append(screen_x)
                x = np.stack(stacked_x, axis=-1)
    return np.mean(scores)


class MemoryBuffer:
    "An experience replay buffer using numpy arrays"
    def __init__(self, length, screen_shape, action_shape):
        self.length = length
        self.screen_shape = screen_shape
        self.action_shape = action_shape
        shape = (length,) + screen_shape
        self.screens_x = np.zeros(shape, dtype=np.uint8) # starting states
        self.screens_y = np.zeros(shape, dtype=np.uint8) # resulting states
        shape = (length,) + action_shape
        self.actions = np.zeros(shape, dtype=np.uint8) # actions
        self.rewards = np.zeros((length,1), dtype=np.uint8) # rewards
        self.terminals = np.zeros((length,1), dtype=np.bool) # true if resulting state is terminal
        self.terminals[-1] = True
        self.index = 0 # points one position past the last inserted element
        self.size = 0 # current size of the buffer
    
    def append(self, screenx, a, r, screeny, d):
        self.screens_x[self.index] = screenx
        #plt.imshow(screenx)
        #plt.show()
        #plt.imshow(self.screens_x[self.index])
        #plt.show()
        self.actions[self.index] = a
        self.rewards[self.index] = r
        self.screens_y[self.index] = screeny
        self.terminals[self.index] = d
        self.index = (self.index+1) % self.length
        self.size = np.min([self.size+1,self.length])
    
    def stacked_frames_x(self, index):
        im_deque = deque(maxlen=4)
        pos = index % self.length
        for i in range(4): # todo
            im = self.screens_x[pos]
            im_deque.appendleft(im)
            test_pos = (pos-1) % self.length
            if self.terminals[test_pos] == False:
                pos = test_pos
        return np.stack(im_deque, axis=-1)
    
    def stacked_frames_y(self, index):
        im_deque = deque(maxlen=4)
        pos = index % self.length
        for i in range(4): # todo
            im = self.screens_y[pos]
            im_deque.appendleft(im)
            test_pos = (pos-1) % self.length
            if self.terminals[test_pos] == False:
                pos = test_pos
        return np.stack(im_deque, axis=-1)
    
    def minibatch(self, size):
        #return np.random.choice(self.data[:self.size], size=sz, replace=False)
        indices = np.random.choice(self.size, size=size, replace=False)
        x = np.zeros((size,)+self.screen_shape+(4,))
        y = np.zeros((size,)+self.screen_shape+(4,))
        for i in range(size):
            x[i] = self.stacked_frames_x(indices[i])
            y[i] = self.stacked_frames_y(indices[i])
        return x, self.actions[indices], self.rewards[indices], y, self.terminals[indices]
        
game = FlappyBird()
p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=True, display_screen=False)
p.init()

def model_stats(rewards, scores):
    print("Average cumulated rewards : {:.2f}".format(np.mean([x for x in rewards])))
    print("Maximum cumulated reward : {:.2f}".format(np.max(rewards)))
    print("Average score : {:.2f}".format(np.mean([x for x in scores])))
    print("Maximum score : {:.2f}".format(np.max(scores)))

# initialize state and replay memory
rewards =[]
scores = []
cumr = 0
score = 0
p.reset_game()
screen_x = process_screen(p.getScreenRGB())
stacked_x = deque([screen_x, screen_x, screen_x, screen_x], maxlen=4)
x = np.stack(stacked_x, axis=-1)
replay_memory = MemoryBuffer(replay_memory_size, (80,80), (1,))
# initial state for evaluation
Xtest = np.array([x])
scoreQ = np.zeros((nb_epochs))
scoreMC = np.zeros((nb_epochs))

# Deep Q-learning with experience replay
for step in range(total_steps):
    #clear_output(wait=True)
    if step%1000 == 0:
        print('step',step,'out of',total_steps)
        if step != 0:
            model_stats(rewards, scores)
    # evaluation
    if(step%10000 == 0):
        if step != 0:
            dqn.save("pixel_bird.dqf")
        # evaluation of initial state
    # action selection
    if np.random.rand() < epsilon(step):
        a = np.random.randint(0,2)
    else:
        a = greedy_action(dqn, x)
    # step
    r = p.act(p.getActionSet()[a])
    r = clip_reward(r)
    cumr += r
    if r == 1:
        score += 1
    screen_y = process_screen(p.getScreenRGB())
    replay_memory.append(screen_x, a, r, screen_y, p.game_over())
    # train
    if step>mini_batch_size:
        X,A,R,Y,D = replay_memory.minibatch(mini_batch_size)
        QY = dqn.predict(Y)
        QYmax = QY.max(1).reshape((mini_batch_size,1))
        update = R + gamma * (1-D) * QYmax
        QX = dqn.predict(X)
        QX[np.arange(mini_batch_size), A.ravel()] = update.ravel()
        dqn.train_on_batch(x=X, y=QX)
    # prepare next transition
    if p.game_over():
        # restart episode
        rewards.append(cumr)
        #print('cumulated rewards : {:.2f}',.format(cumr))
        cumr = 0
        scores.append(score)
        #print('score : {:d}',.format(score))
        score = 0
        p.reset_game()
        screen_x = process_screen(p.getScreenRGB())
        stacked_x = deque([screen_x, screen_x, screen_x, screen_x], maxlen=4)
        x = np.stack(stacked_x, axis=-1)
    else:
        # keep going
        screen_x = screen_y
        stacked_x.append(screen_x)
        x = np.stack(stacked_x, axis=-1)
        
plt.plot(rewards)
model_stats(rewards, scores)
