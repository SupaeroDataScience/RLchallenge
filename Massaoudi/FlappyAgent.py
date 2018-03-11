import numpy as np
from keras.models import load_model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
import random
from ple.games.flappybird import FlappyBird
from ple import PLE
model_test=load_model(r"model.h5")
def FlappyPolicy(state, screen):
    q=model_test.predict(np.array(list(state.values())).reshape(1,8))
    max_Q_index = np.argmax(q)
    action=[119,None][max_Q_index]
    return action

class Agent:
    def __init__(self,nb_games=20000,gamma=0.99,epsilon=1,batchsize=30,buffer=80):
        self.nb_games = nb_games
        self.gamma = gamma # discount factor
        self.epsilon = epsilon # epsilon-greddy
        self.batchSize = batchsize # mini batch size
        self.buffer = buffer

    def build_model(self):
        model = Sequential()
        model.add(Dense(500, init='lecun_uniform', input_shape=(8,)))
        model.add(Activation('relu'))
        model.add(Dense(2, init='lecun_uniform'))
        model.add(Activation('linear'))
        model.compile(loss='mse', optimizer=Adam(lr=1e-5))
        return model
    def train(self):
        update=0 
        replay = [] # init vector buffer
        h=0 # current size of the vector buffer
        game_number=0
        model=self.build_model()
        game = FlappyBird(graphics="fancy")
        p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=False, display_screen=True)
        p.reset_game()
        actions=[119,None]
        while(True):
            p.reset_game()
            state = game.getGameState()
            while(not p.game_over()):
                qval = model.predict(np.array(list(state.values())).reshape(1,8))
                if (random.random() < self.epsilon): # choose randomly an exploration/exploitation strategy
                    action=actions[np.random.randint(0,2)] # exploration
                else: #choose best action from Q(s,a) values
                    action = actions[np.argmax(qval)] # exploitation
                #Take action, observe new state S'
                #Observe reward and modify it
                reward=p.act(action)
                if reward >0:
                    reward=15*reward
                if reward <0:
                    reward=2*reward
                new_state = game.getGameState()
                terminal=p.game_over() #boolean terminal state : True if the game is over
                
                #Experience replay storage
                if (len(replay) < self.buffer): #if buffer not filled, add to it                
                    replay.append((state, action, reward, new_state,terminal))
                else: #if buffer full, overwrite old values
                    if (h < (self.buffer-1)):
                        h += 1
                    else:
                        h = 0
                    replay[h] = (state, action, reward, new_state,terminal)
                    #randomly sample our experience replay memory
                    minibatch = random.sample(replay, self.batchSize)
                    X_train = []
                    y_train = []
                    for memory in minibatch:
                        s1, action, reward, s2,terminal = memory
                        old_qval = model.predict(np.array(list(s1.values())).reshape(1,8), batch_size=1)
                        newQ = model.predict(np.array(list(s2.values())).reshape(1,8), batch_size=1)
                        maxQ = np.max(newQ)
                        y = np.zeros((1,2))
                        y[:] = old_qval[:]
                        #calculate the update value for terminal and non terminal state
                        if not terminal: #non-terminal state
                            update = (reward + (self.gamma * maxQ))
                        else: #terminal state
                            update = reward
                        action_index= 0 if action==119 else 1
                        y[0][action_index] = update
                        X_train.append(np.array(list(s1.values())).reshape(len(s1),))
                        y_train.append(np.array(y).reshape(2,))
        
                    X_train = np.array(X_train)
                    y_train = np.array(y_train)
                    model.fit(X_train, y_train, batch_size=self.batchSize, nb_epoch=1, verbose=0)
                state=new_state
            # update exploitation / exploration strategy
            game_number+=1
            print("Game {}, epsilon= {}".format(game_number,self.epsilon))
            if self.epsilon > 0.2:
                self.epsilon -= (1.0/(self.nb_games))
            # save the model every 1000 epochs
            if game_number%1000 == 0:
                model.save("model{}.h5".format(game_number//1000))
if __name__=="__main__":
    agent=Agent()
    agent.train()