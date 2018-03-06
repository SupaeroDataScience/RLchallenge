# -*- coding: utf-8 -*-

import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation
import random
from IPython.display import clear_output
# You're not allowed to change this file
from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np
import time

# This function will be used to test the current model over 10 games
def test():
    game2 = FlappyBird()
    p2 = PLE(game2, fps=30, frame_skip=1, num_steps=1, force_fps=True, display_screen=False)
    p2.init()
    reward = 0.0

    nb_games = 10
    cumulated = np.zeros((nb_games))
    for i in range(nb_games):
        p2.reset_game()
    
        while(not p2.game_over()):
            state = game2.getGameState()
            screen = p2.getScreenRGB()
            action= FlappyPolicy(state, screen)
        
            reward = p2.act(action)
            cumulated[i] = cumulated[i] + reward
    return np.mean(cumulated)

# This function returns the best action to perform with the current model
def FlappyPolicy(state, screen):
    batchSize = 2

    qval = model.predict(np.array(list(state.values())).reshape(1,8), batch_size=batchSize) 
    
    qval_av_action = qval[0]
    action = (np.argmax(qval_av_action))*119
    return action


# Model initialisation
# The model is composed of a single dense layer of size 800
# The learning rate is set to a low value to avoid having the same output for every input
model=Sequential()

model.add(Dense(800,input_shape=(8,), init='lecun_uniform'))
model.add(Activation('relu'))

model.add(Dense(2, init='lecun_uniform'))
model.add(Activation('linear'))

model.compile(loss="mse", optimizer=keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8))


# The number of episode needed to achive the best model was 37000.
# I used the experience replay with a buffer size of 40 and a batchsize of 2

ep = 50000
gamma = 0.9 # discount factor
epsilon = 1 # epsilon-greddy

update=0
batchSize = 2 # mini batch size
buffer = 40 
replay = [] # init vector buffer
h=0 # current size of the vector buffer
record = []

# Game initialization
game = FlappyBird()
p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=True, display_screen=False)
p.init()

# We will train over 50000 episodes
for i in range(ep):
    p.reset_game()
    
    while(not p.game_over()):
        state = game.getGameState()   
        qval = model.predict(np.array(list(state.values())).reshape(1,8), batch_size=batchSize) 

        if (random.random() < epsilon): # exploration exploitation strategy    
            action = np.random.randint(2)
        else: #choose best action from Q(s,a) values
            action = np.argmax(qval)
        # Actual play with the chosen action
        reward = p.act(action*119)
        
        # We change the reward value when we loose to help the model
        if reward == -5:
            reward = -100
            
        newstate= game.getGameState()
        # if the buffer is not full, we fill it, otherwise we will replace one row with a new one
        if(len(replay)<buffer):
            replay.append((state, action, reward, newstate ))
        else:
            if (h < (buffer-1)):
                h=h+1
            else:
                h=0
            replay[h] = (state, action, reward, newstate )
            
            # We use a simple random sample to create the minibatch
            minibatch = random.sample(replay, batchSize)
            
            x_train=[]
            y_train=[]
            
            # training over the minibatch
            
            for memory in minibatch:
                oldstate, action, reward, newstate = memory
                old_qval = model.predict(np.array(list(oldstate.values())).reshape(1,8), batch_size=1)
                newQ = model.predict(np.array(list(newstate.values())).reshape(1,8), batch_size=1)
                maxQ = np.max(newQ)
                y = np.zeros((1,2))
                y[:] = old_qval[:]
                
                if reward != -100: #non-terminal state
                    update = (reward + (gamma * maxQ))
                else:
                    update = reward
                y[0][action] = update
                x_train.append(np.array(list(oldstate.values())).reshape(1,8))
                y_train.append(y)
                
                
            x_train = np.array(x_train).reshape(batchSize,8)
            y_train = np.array(y_train).reshape(batchSize,2)
            model.fit(x_train, y_train, batch_size=batchSize, epochs=1, verbose=1)
            state = newstate
            
        
        print("episode : ", i)
        print('scores :', record)        
        clear_output(wait=True)
           
    # Decreasing epsilon over 20 000 epochs to 0.05 and then training with this value
    if epsilon > 0.05:
        epsilon -= (1.0/20000)
        
    if i%1000 == 0:
        model.save("models\model_dql_simple"+str(i) +".dqf")
        # we test the current model
        record.append(test())    
        time.sleep(10)
        
    