
from ple.games.flappybird import FlappyBird
from ple import PLE
from keras import optimizers
from keras.models import Sequential
from keras.layers.core import Dense, Activation
import numpy as np
import random


model = Sequential()


model.add(Dense(500, init='lecun_uniform', input_shape=(8,)))

model.add(Activation('relu'))
model.add(Dense(2, init='lecun_uniform'))
model.add(Activation('linear'))
model.compile(loss='mse', optimizer=optimizers.Adam(1e-4))


epochs = 28000
gamma = 0.99 # discount factor
epsilon = 0.7 # epsilon-greddy
batchSize = 256 # mini batch size

jeu = FlappyBird()
p= PLE(jeu, fps=30, frame_skip=1, num_steps=1,force_fps=True, display_screen=False)
p.init()

i=0

for i in range(epochs):
    p.reset_game()
    state = jeu.getGameState()
    state = np.array(list(state.values()))
    while(not jeu.game_over()):
        
       
        
        qval = model.predict(state.reshape(1,len(state)), batch_size=batchSize) #Learn Q (Q-learning) / model initialise avant (neural-network)
        if (random.random() < epsilon): # exploration exploitation strategy    
            action = np.random.randint(0,2)
#             print(action)
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
            reward = -1000
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
        model.fit(state.reshape(1, len(state)), y, batch_size=batchSize, nb_epoch=3, verbose=0)
        state = new_state
        
    
    # update exploitation / exploration strategy
    if epsilon > 0.2:
        epsilon -= (1.0/epochs)



model.save("model.dqf")
