import numpy as np
from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np
from keras import optimizers
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop, sgd
from keras.layers.recurrent import LSTM
import numpy as np
import random
import sys
from sklearn.preprocessing import StandardScaler as scl
import time
model = Sequential()

file_path = "test_part_"


model.add(Dense(512, init='lecun_uniform', input_shape=(8,)))

model.add(Activation('relu'))
model.add(Dense(2, init='lecun_uniform'))
model.add(Activation('linear'))
model.compile(loss='mse', optimizer=optimizers.Adam(lr=1e-4))

gamma = 0.99 # discount factor
epsilon = 1 # epsilon-greddy
batchSize = 256 # mini batch size

jeu = FlappyBird()
p= PLE(jeu, fps=30, frame_skip=1, num_steps=1,force_fps=True, display_screen=True)
p.init()

i=0

while (True):
    p.reset_game()
    state = jeu.getGameState()
    state = np.array(list(state.values()))
    while(not jeu.game_over()):
        
       
        
        qval = model.predict(state.reshape(1,len(state)), batch_size=batchSize) #Learn Q (Q-learning) / model initialise avant (neural-network)
        if (random.random() < epsilon): # exploration exploitation strategy    
            action = np.random.randint(0,2)
        else: #choose best action from Q(s,a) values
            qval_av_action = [-9999]*2
            
            for ac in range(0,2):
                qval_av_action[ac] = qval[0][ac]
            action = (np.argmax(qval_av_action))
        #Take action, observe new state S'
        #Observe reward
        reward = p.act(119*action)
        if reward == 1:
            reaward = 1
        elif reward == -5:
            reward = -500
        new_state = jeu.getGameState()
        new_state = np.array(list(new_state.values()))
        # choose new reward values
        
            
        #Get max_Q(S',a)
        newQ = model.predict(new_state.reshape(1,len(state)), batch_size=batchSize)
        maxQ = np.max(newQ)
        y = np.zeros((1,2))
        y[:] = qval[:]
        if reward != -5: #non-terminal state
            update = (reward + gamma * maxQ)
        else:
            update = reward
        y[0][action] = update
        print("Game #: %s" % (i,))
        model.fit(state.reshape(1, len(state)), y, batch_size=batchSize, nb_epoch=2, verbose=0)
        state = new_state
        
    
    # update exploitation / exploration strategy
    if epsilon > 0.1:
        epsilon -= (1.0/10000)

    # save the model every 1000 epochs
    if i==100:
        model.save(file_path+"0.dqf")
    if i%1000 == 0 and i!=0:
        model.save(file_path+str(i/1000)+".dqf")
        time.sleep(60)
    if i == 100000:
        break
        
    i=i+1
model.save(file_path+"final.dqf")
