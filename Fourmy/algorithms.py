import os
import time
import pickle
import random
import numpy as np
from collections import deque
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam

from skimage.color import rgb2gray
from skimage.transform import resize
# from sklearn.preprocessing import StandardScaler

from ple import PLE

from utils import (myround, delete_files, init_train, print_scores,
                   update_epsilon)


ACTIONS = [None, 119]
STATES = [
    'next_next_pipe_top_y', 'next_pipe_top_y', 'next_pipe_bottom_y',
    'next_next_pipe_bottom_y', 'next_next_pipe_dist_to_player',
    'next_pipe_dist_to_player', 'player_y',  'player_vel'
]
STATE_BOUNDS = np.array([
    [0., 0., 0., 0., 0., 0., 0., -8.],
    [387., 387., 387., 387., 427., 283., 387., 10.],
    ])
NB_LAST_SCREENS = 4
SIZE_IMG = (80, 80)             # Size of the processed image


# Note: if you want to see you agent act in real time, set force_fps to False.
# But don't use this setting for learning, just for display purposes.

# 1) In s, choose a (GLIE actor)
# 2) Observe r, s′
# 3) Temporal difference:
# delta = r + self.GAMMA*maxa′Q(s′,a′)−Q(s,a)δ=r+γmaxa′Q(s′,a′)−Q(s,a)
# 4) Update Q :  Q(s,a) ← Q(s,a) + αδQ(s,a) ← Q(s,a)+αδ
# 5) s <- s′


class ReplayMemory:

    def __init__(self):
        self.buff = deque([], REPLAY_MEMORY_SIZE)

    def append(self, screen, act, screen_new, reward):
        self.buff.append(screen, act, screen_new, reward)

    def last_screens(self, buff_index):
        buff_index = max(buff_index, NB_LAST_SCREENS-1)
        last_screens = []
        for i in range(buff_index-NB_LAST_SCREENS, buff_index+1):
            last_screens.append(self.buff[i][0])
        return np.stack(last_screens, axis=-1)

    def last_screens_new(self, buff_index):
        # TODO: refactoring
        buff_index = max(buff_index, NB_LAST_SCREENS-1)
        last_screens_new = []
        for i in range(buff_index-NB_LAST_SCREENS, buff_index+1):
            last_screens_new.append(self.buff[i][0])
        return np.stack(last_screens, axis=-1)

    def minibatch(self):
        indices = np.random.choice(REPLAY_MEMORY_SIZE,
                                   self.BATCH_SIZE, replace=False)
        last_screens = np.zeros((BATCH_SIZE,) + SIZE_IMG + (NB_LAST_SCREENS,))
        last_screens_new = np.zeros((BATCH_SIZE,) + SIZE_IMG + (NB_LAST_SCREENS,))
        actions = np.zeros(BATCH_SIZE)
        rewards = np.zeros(BATCH_SIZE)
        for i, buff_index in enumerate(indices):
            last_screens[i] = self.last_screens(buff_index)
            last_screens_new[i] = self.last_screens_new(buff_index)

        return last_screens, action, last_screens_new, rewards


class DeepQLearning:

    NB_FRAMES = 1000000
    SAVE_FREQ = NB_FRAMES // 5
    EPS_UPDATE_FREQ = 10000
    SCORE_FREQ = 100
    STEPS_TARGET = 2500

    REPLAY_MEMORY_SIZE = 1000
    TRAIN_FREQ = 5
    BATCH_SIZE = 32

    GAMMA = 0.9  # discount factor
    UP_PROBA = 0.5
    EPS0 = 0.2
    EPS_RATE = 4
    ALPHA = 0.2  # learning rate

    NB_TEST = 100

    DATA_DIREC = 'data/DQL/'

    def __init__(self, game):
        self.game = game
        self.p = PLE(self.game, fps=30, frame_skip=1, num_steps=1,
                     force_fps=True, display_screen=display)
        self.epsilon = self.EPS0
        self.model = self.create_model()
        self.model_target = self.create_model()
        self.replay_memory = ReplayMemory(REPLAY_MEMORY_SIZE)

    def get_qvals(self, last_screens):
        return self.model.predict(last_screens)

    def process_screen(self, scree):
        screen_cut = screen[60:, 25:310, :]
        screen_grey = 256 * (rgb2gray(screen_cut))
        return resize(screen_grey, SIZE_IMG, mode='constant')

    def greedy_action(self, qvals, epsilon):
        if random.random() < epsilon:  # exploration
            return 1 if random.random() < self.UP_PROBA else 0
        else:
            return np.argmax(qvals)

    def train(self, scratch):
        fname = None
        if not scratch:
            fname = self.load()
        f0, step, nb_save, nb_games = init_train(fname, self.DATA_DIREC)

        eps_tau = (self.NB_FRAMES - f0)//self.EPS_RATE
        scores = []
        while step < self.NB_FRAMES:
            if len(scores) == self.SCORE_FREQ:
                print_scores(scores, self.SCORE_FREQ)
                scores = []

            self.p.reset_game()
            _ = self.game.getGameState()
            screen = self.process_screen(self.p.getScreenRGB())
            last_screens_buff = deque([screen]*4, maxlen=NB_LAST_SCREENS)
            x = np.stack(last_screens_buff, axis=-1)

            gscore = 0
            nb_games += 1
            while not self.p.game_over():
                step += 1
                if step != 0 and (step % self.SAVE_FREQ) == 0:
                    self.save(chr(97+nb_save) + '_' + str(step) +
                              '_' + str(nb_games))
                    nb_save += 1
                if step != 0 and (step % self.EPS_UPDATE_FREQ) == 0:
                    self.epsilon = update_epsilon(step, f0, self.EPS0,
                                                  eps_tau, self.NB_FRAMES)
                    print('WEIGHTS ABS MEAN')
                    print(abs(np.mean(self.model.get_weights()[0], axis=1)))

                # 1) In s, choose a (GLIE actor)
                qvals = self.get_qvals(last_screens)
                act = self.greedy_action(qvals, self.epsilon)

                # 2) Observe r, s′
                bare_reward = self.p.act(ACTIONS[act])
                reward = self.reward_engineering(reward)
                screen_new = self.process_screen(self.p.getScreenRGB())

                # update replay_memory
                replay_memory.append((screen, act, screen_new, reward))
                if len(replay_memory) == REPLAY_MEMORY_SIZE:
                    # build minibatch
                    ls, actions, ls_new, r = self.replay_memory.minibatch()
                    qval_new = self.model_target(ls_new)
                    print(qval_new.shape)










                last_screens_buff.append(screen_new)

    def reward_engineering(self, reward):
        return reward

    def create_model(self, img_size_x, img_size_y):
        input_shape = (img_size_x, img_size_y, 4)
        model = Sequential()
        model.add(Conv2D(filters=16, kernel_size=(8, 8), strides=4,
                         activation="relu", input_shape=input_shape))
        model.add(Conv2D(filters=32, kernel_size=(4, 4), strides=2,
                         activation="relu"))
        model.add(Flatten())
        model.add(Dense(units=256, activation="relu"))
        model.add(Dense(units=len(ACTIONS), activation="linear"))
        model.compile(optimizer=Adam(lr=params.LEARNING_RATE),
                      loss="mean_squared_error")
        return model

    def save(self, name):
        # DQN + DQN target



class FeaturesNeuralQLearning:

    NB_FRAMES = 1000000
    SAVE_FREQ = NB_FRAMES // 5
    EPS_UPDATE_FREQ = 10000
    SCORE_FREQ = 100

    BUFFER_SIZE = 1000
    TRAIN_FREQ = 5
    BATCH_SIZE = 32

    GAMMA = 0.9  # discount factor
    UP_PROBA = 0.5
    EPS0 = 0.2
    EPS_RATE = 4
    ALPHA = 0.2  # learning rate

    DATA_DIREC = 'data/FNQL/'

    def __init__(self, game, display):
        self.game = game
        self.p = PLE(self.game, fps=30, frame_skip=1, num_steps=1,
                     force_fps=True, display_screen=display)
        self.epsilon = self.EPS0
        self.replay_memory = deque([], self.BUFFER_SIZE)
        self.model = self.create_model()

        # self.scaler = StandardScaler().fit(STATE_BOUNDS)

    def get_qvals(self, state):
        state_arr = self.state_to_arr(state)
        return self.model.predict(state_arr, batch_size=self.BATCH_SIZE)

    def greedy_action(self, qvals, epsilon):
        if random.random() < epsilon:  # exploration
            return 1 if random.random() < self.UP_PROBA else 0
        else:
            return np.argmax(qvals)

    def train(self, scratch):
        fname = None
        if not scratch:
            fname = self.load()
        f0, step, nb_save, nb_games = init_train(fname, self.DATA_DIREC)

        eps_tau = (self.NB_FRAMES - f0)//self.EPS_RATE
        scores = []
        while step < self.NB_FRAMES:
            if len(scores) == self.SCORE_FREQ:
                print_scores(scores, self.SCORE_FREQ)
                scores = []

            self.p.reset_game()
            state = self.game.getGameState()
            state_arr = self.state_to_arr(state)
            # state_arr = self.scaler.transform(state_arr.reshape(1, -1))
            gscore = 0
            nb_games += 1
            while not self.p.game_over():
                step += 1
                if step != 0 and (step % self.SAVE_FREQ) == 0:
                    self.save(chr(97+nb_save) + '_' + str(step) +
                              '_' + str(nb_games))
                    nb_save += 1
                if step != 0 and (step % self.EPS_UPDATE_FREQ) == 0:
                    self.epsilon = update_epsilon(step, f0, self.EPS0,
                                                  eps_tau, self.NB_FRAMES)
                    print('WEIGHTS ABS MEAN')
                    print(abs(np.mean(self.model.get_weights()[0], axis=1)))

                # 1) In s, choose a (GLIE actor)
                qvals = self.get_qvals(state)
                act = self.greedy_action(qvals, self.epsilon)

                # 2) Observe r, s′
                bare_reward = self.p.act(ACTIONS[act])
                new_state = self.game.getGameState()
                new_state_arr = self.state_to_arr(state)

                self.replay_memory.append((state_arr, act,
                                  bare_reward, new_state_arr))
                if (len(self.replay_memory) == self.BUFFER_SIZE
                   and step % self.TRAIN_FREQ == 0):

                    X_train = []
                    y_train = []

                    # TEST: TRAIN ONLY WITH A SMALL BUFFER BATCH
                    replay_memory_copy = list(self.replay_memory)[:]
                    random.shuffle(replay_memory_copy)
                    for frame in replay_memory_copy[:self.BATCH_SIZE]:
                        s_arr_1, act_x, bare_reward_x, s_arr_2 = frame
                        reward_x = self.reward_engineering(bare_reward_x)
                        old_qval = self.model.predict(s_arr_1, batch_size=1)
                        qval_new = self.model.predict(s_arr_2, batch_size=1)
                        max_qval = np.max(qval_new)
                        # terminal state
                        if bare_reward < 0:
                            delta = reward_x
                        else:
                            delta = reward_x + self.GAMMA * max_qval  # WTF!!!
                            # delta = reward_x + self.GAMMA*max_qval - old_qval[0][act_x]
                        y = np.zeros((1, len(ACTIONS)))
                        y[0][:] = old_qval[0][:]
                        y[0][act_x] = old_qval[0][act_x] + self.ALPHA*delta
                        X_train.append(s_arr_1.reshape(len(STATES),))
                        y_train.append(y.reshape(len(ACTIONS),))

                    X_train = np.array(X_train)
                    y_train = np.array(y_train)
                    self.model.fit(X_train, y_train,
                                   batch_size=self.BATCH_SIZE,
                                   epochs=2, verbose=False)

                # 5) s <- s'
                state = new_state
                state_arr = new_state_arr

                if bare_reward > 0:
                    gscore += 1
            scores.append(gscore)

        self.save(chr(97+nb_save)+'_'+str(step)+'_' + str(nb_games))

    def reward_engineering(self, reward):
        # TODO: should be done with reward_values dict
        if reward < 0:
            return -100
        return reward

    def save(self, name):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(os.path.join(self.DATA_DIREC, name+'.json'), 'w') as f:
            f.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(os.path.join(self.DATA_DIREC, name+'.h5'))
        print('Saved model to disk', name)

    def load(self, name=None):
        if name is None:
            files = os.listdir(self.DATA_DIREC)
            if len(files) == 0:
                return None
            files_without_ext = [f.split('.')[0] for f in files]
            name = max(files_without_ext)

            with open(os.path.join(self.DATA_DIREC, name+'.json'), 'r') as f:
                loaded_model_json = f.read()
            self.model = model_from_json(loaded_model_json)
            # load weights into new model
            self.model.load_weights(os.path.join(self.DATA_DIREC, name+'.h5'))

            print('###########')
            print('Files loaded: ', name)
            print('###########')
            return name

    def create_model(self, size1=150, size2=150):
        model = Sequential()
        model.add(Dense(size1, kernel_initializer='lecun_uniform',
                  input_shape=(len(STATES),)))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(size2, kernel_initializer='lecun_uniform'))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(len(ACTIONS), kernel_initializer='lecun_uniform'))
        model.add(Activation('linear'))
        model.compile(optimizer=Adam(lr=1e-4, loss="mean_squared_error"))
        return model

    def state_to_arr(self, state):
        return np.array([state[feature] for feature in STATES])\
                 .reshape(1, len(STATES))


class FeaturesLambdaSarsa:
    STATES_USED = [
        'next_pipe_top_y',
        'next_pipe_dist_to_player',
        'player_y',
        'player_vel',
    ]
    ACTIONS = [None, 119]

    NB_FRAMES = 4000000
    SAVE_FREQ = NB_FRAMES // 10
    EPS_UPDATE_FREQ = 10000
    SCORE_FREQ = 100

    GAMMA = 0.9  # discount factor
    UP_PROBA = 0.1
    EPS0 = 0.4
    LAMBDA = 0.8
    ALPHA = 0.2
    # TODO: remove
    SIZE_FIFO = None

    DATA_DIREC = 'data/FLS/'

    def __init__(self, game, display):
        self.game = game
        self.p = PLE(self.game, fps=30, frame_skip=1, num_steps=1,
                     force_fps=True, display_screen=display)
        self.epsilon = self.EPS0  # epsilon-greddy
        # (feature1, feature1, feature1): [qval_a1, qval_a2]
        self.Q = {}

    def get_qvals(self, state):
        state_tp = self.discretize(state)
        if state_tp in self.Q:
            return self.Q[state_tp]
        else:
            return [0, 0]

    def greedy_action(self, qvals, epsilon):
        if random.random() < epsilon or qvals == [0, 0]:
            return  1 if random.random() < self.UP_PROBA else 0
        else:
            return np.argmax(qvals)

    def train(self, scratch=True):
        t1 = time.time()
        fname = None
        if not scratch:
            fname = self.load()
        f0, step, nb_save, nb_games = init_train(fname, self.DATA_DIREC)

        eps_tau = (self.NB_FRAMES - f0)//8

        scores = []
        while step < self.NB_FRAMES:
            if len(scores) == self.SCORE_FREQ:
                print('States visited:', len(self.Q))
                print_scores(scores, self.SCORE_FREQ)
                scores = []
            self.p.reset_game()
            state = self.game.getGameState()
            state_tp = self.discretize(state)
            if state_tp not in self.Q:
                self.Q[state_tp] = [0, 0]

            act = 1
            episode = deque([], self.SIZE_FIFO)
            elig = {}
            gscore = 0
            nb_games += 1
            while not self.p.game_over():
                step += 1
                if step != 0 and (step % self.SAVE_FREQ) == 0:
                    self.save('Q_' + chr(97+nb_save) + '_' + str(step) +
                              '_' + str(nb_games) + '.p')
                    nb_save += 1
                if step != 0 and (step % self.EPS_UPDATE_FREQ) == 0:
                    self.epsilon = update_epsilon(step, f0, self.EPS0,
                                                  eps_tau, self.NB_FRAMES)
                # 1) Observe r, s′
                bare_reward = self.p.act(ACTIONS[act])
                reward = self.reward_engineering(bare_reward)
                new_state = self.game.getGameState()
                new_state_tp = self.discretize(new_state)

                # 2) Choose a′ (GLIE actor) using Q
                if new_state_tp not in self.Q:
                    self.Q[new_state_tp] = [0, 0]
                qvals = self.get_qvals(new_state)
                new_act = self.greedy_action(state, self.epsilon)

                # 3) Temporal difference:  δ=r+γQ(s′,a′)−Q(s,a)
                delta = reward + self.GAMMA*self.Q[new_state_tp][new_act] - self.Q[state_tp][act]

                # 4) Update Q
                episode.append((state_tp, act))
                elig[(state_tp, act)] = 1
                for (state_tp_ep, act_ep) in episode:
                    self.Q[state_tp_ep][act_ep] += (
                            self.ALPHA * delta * elig[(state_tp_ep, act_ep)])
                    elig[(state_tp_ep, act_ep)] *= self.LAMBDA

                # 5) s<-s', a<-a'
                state = new_state
                state_tp = new_state_tp
                act = new_act

                if bare_reward > 0:
                    gscore += 1

            scores.append(gscore)

        t2 = time.time()
        # Unicode code point of a: 97
        self.save('Q_' + chr(97+nb_save) + '_' + str(step) +
                  '_' + str(nb_games) + '.p')
        print()
        print('Number of played games:', nb_games)
        print('Training completed in', (t2 - t1)/60, 'minutes')
        print()

    def discretize(self, state):
        # approximate as a lower pipe
        # ~ 200/x states
        state['next_pipe_top_y'] = myround(state['next_pipe_top_y'], 20)
        # ~ 200/x states
        state['next_pipe_dist_to_player'] = myround(state['next_pipe_dist_to_player'], 20)
        # ~400/x states
        state['player_y'] = myround(state['player_y'], 20)
        # 17 states
        state['player_vel'] = myround(state['player_vel'], 1)
        return tuple(state[feature] for feature in self.STATES_USED)

    def reward_engineering(self, reward):
        return reward

    def save(self, name):
        with open(os.path.join(self.DATA_DIREC, name), 'bw') as f:
            pickle.dump(self.Q, f)
        print('Saved Q to disk', name)

    def load(self, name=None):
        if name is None:
            files = os.listdir(self.DATA_DIREC)
            try:
                name = max(files)
            except ValueError as e:
                print('\nError: No file in ' + self.DATA_DIREC)
                raise e
        with open(os.path.join(self.DATA_DIREC, name), 'rb') as f:
            self.Q = pickle.load(f)
        print('###########')
        print('File loaded: ', name)
        print('###########')
        return name
