import os
import time
import pickle
import random
import numpy as np
from collections import deque
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation

from ple.games.flappybird import FlappyBird
from ple import PLE


def myround(x, base):
    return int(base * round(float(x)/base))


def rounddown(x, base):
    return int(x - (x % base))


def roundup(x, base):
    return int(x - (x % base) + base)


def delete_files(folder_path):
    for the_file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(e)


# Note: if you want to see you agent act in real time, set force_fps to False.
# But don't use this setting for learning, just for display purposes.

# 1) In s, choose a (GLIE actor)
# 2) Observe r, s′
# 3) Temporal difference:
# delta = r + self.GAMMA*maxa′Q(s′,a′)−Q(s,a)δ=r+γmaxa′Q(s′,a′)−Q(s,a)
# 4) Update Q :  Q(s,a) ← Q(s,a) + αδQ(s,a) ← Q(s,a)+αδ
# 5) s <- s′


class FeaturesNeuralQLearning:
    STATES = [
        'next_next_pipe_top_y', 'next_pipe_top_y', 'next_pipe_bottom_y',
        'next_next_pipe_bottom_y', 'next_next_pipe_dist_to_player',
        'next_pipe_dist_to_player', 'player_y',  'player_vel'
    ]
    # STATES = [
    #     'next_pipe_bottom_y',
    #     'next_pipe_dist_to_player',
    #     'player_y',
    #     'player_vel',
    # ]
    ACTIONS = [None, 119]

    NB_FRAMES = 2000000
    SAVE_FREQ = NB_FRAMES // 10
    EPS_UPDATE_FREQ = 10000
    SCORE_FREQ = 100

    BUFFER_SIZE = 100
    BATCH_SIZE = 32

    GAMMA = 0.9  # discount factor
    UP_PROBA = 0.1
    EPS0 = 0.5
    EPS_T = NB_FRAMES//6
    ALPHA = 0.2  # learning rate
    # LAMBDA = 0.8
    # SIZE_FIFO = 15

    NB_TEST = 100

    DATA_DIREC = 'data/FNQL/'

    def __init__(self):
        # COMMON
        self.game = FlappyBird()
        self.p = PLE(self.game, fps=30, frame_skip=1, num_steps=1,
                     force_fps=True, display_screen=True)
        self.epsilon = self.EPS0
        self.buff = []  # init vector buffer
        # COMMON

        self.buffer_idx = 0
        self.model = self._create_model()

    def play(self, n=1):
        self.p = PLE(self.game, fps=30, frame_skip=1, num_steps=1,
                     force_fps=False, display_screen=True)
        for _ in range(n):
            self.p.reset_game()
            while not self.p.game_over():
                state = self.game.getGameState()
                state_arr = self.state_to_arr(state, self.STATES)
                qval = self.model.predict(state_arr, batch_size=self.BATCH_SIZE)
                act = np.argmax(qval)
                self.p.act(self.ACTIONS[act])
                state = self.game.getGameState()
                state_arr = self.state_to_arr(state, self.STATES)

    def play_SARSA(self, n=1):
        self.p = PLE(self.game, fps=30, frame_skip=1, num_steps=1,
                     force_fps=False, display_screen=True)
        for _ in range(n):
            self.p.reset_game()
            while not self.p.game_over():
                nb_not_seen = 0
                state = self.game.getGameState()
                state_tp = self.discretize(state)
                if state_tp not in self.Q:
                    nb_not_seen += 1
                    act = 1 if random.random() < self.UP_PROBA else 0
                else:
                    qval = self.Q[state_tp]
                    act = np.argmax(qval)
                self.p.act(self.ACTIONS[act])
            print('Not seen in this game:', nb_not_seen)

    def train(self):
        # delete_files()
        t1 = time.time()
        nb_train = 0
        curr_frame = 0
        nb_save = 0
        scores = []
        while curr_frame < self.NB_FRAMES:
            if len(scores) == self.SCORE_FREQ:
                print([(str(s) if s != 0 else '.') for s in scores])
            self.p.reset_game()
            state = self.game.getGameState()
            state_arr = self.state_to_arr(state, self.STATES)

            while not self.p.game_over():
                score = 0
                if curr_frame != 0 and (curr_frame % self.SAVE_FREQ) == 0:
                    self.save_model('model_popo_' + str(nb_save))
                    nb_save += 1
                if curr_frame != 0 and (curr_frame % self.EPS_UPDATE_FREQ) == 0:
                    self.epsilon *= self.EPS_RED
                    print('CURRENT FRAME:', curr_frame,
                          100*curr_frame / self.NB_FRAMES, '%',
                          'EPSILON: ', self.epsilon)
                # 1) In s, choose a (GLIE actor)
                qval = self.model.predict(state_arr, batch_size=self.BATCH_SIZE)
                if random.random() < self.epsilon:  # exploration
                    rdm = random.random()
                    act = 1 if rdm < self.UP_PROBA else 0
                else:
                    act = np.argmax(qval)

                # 2) Observe r, s′
                reward = self.p.act(self.ACTIONS[act])
                new_state = self.game.getGameState()
                new_state_arr = self.state_to_arr(state, self.STATES)

                if len(self.buff) < self.BUFFER_SIZE:
                    # No learning until the buffer is full
                    self.buff.append((state_arr, act,
                                      reward, new_state_arr))
                else:
                    nb_train += 1
                    if nb_train == 50:
                        t2 = time.time()
                        print('Time estimated: ',
                              (self.NB_FRAMES*(t2-t1)/50)/3600, 'hours')
                    self.buff[self.buffer_idx] = (state_arr, act,
                                                  reward, new_state_arr)
                    self.buffer_idx = (self.buffer_idx + 1) % self.BUFFER_SIZE

                    X_train = []
                    y_train = []
                    for frame in self.buff:
                        # Get max_Q(S',a)
                        s_arr_1, act_x, reward_x, s_arr_2 = frame
                        old_qval = self.model.predict(s_arr_1, batch_size=1)
                        qval_new = self.model.predict(s_arr_2, batch_size=1)
                        max_qval = np.max(qval_new)
                        delta = reward_x + self.GAMMA*max_qval - old_qval[0][act_x]
                        y = np.zeros((1, len(self.ACTIONS)))
                        y[0][:] = old_qval[0][:]
                        y[0][act_x] = old_qval[0][act_x] + self.ALPHA*delta
                        X_train.append(s_arr_1.reshape(len(self.STATES),))
                        y_train.append(y.reshape(len(self.ACTIONS),))

                    X_train = np.array(X_train)
                    y_train = np.array(y_train)
                    self.model.fit(X_train, y_train,
                                   batch_size=self.BUFFER_SIZE,
                                   epochs=1, verbose=False)

                # 5) s <- s'
                state = new_state
                state_arr = new_state_arr

                # print('REWARD: ', reward)
                if reward > 0:
                    score += 1
                    print('YEAHHHH!')

                curr_frame += 1
            scores.append(score)

    def _create_model(self):
        # Default model used in RL notebook 4
        model = Sequential()
        model.add(Dense(150, init='lecun_uniform',
                  input_shape=(len(self.STATES),)))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(150, init='lecun_uniform'))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(len(self.ACTIONS), init='lecun_uniform'))
        model.add(Activation('linear'))
        model.compile(loss='mse', optimizer="rmsprop")
        return model

    def save_model(self, name):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(os.join(self.DATA_DIREC, name + '.json'), 'w') as f:
            f.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(os.path.join(self.DATA_DIREC, name + '.h5'))
        print('Saved model to disk', name)

    def load_model(self, name):
        # load json and create model
        with open(os.path.join(self.DATA_DIREC+name, 'model.json', 'r')) as f:
            loaded_model_json = f.read()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.load_weights(os.path.join(self.DATA_DIREC+name, name + '.json'))
        print("Loaded model from disk")

    def state_to_arr(self, state):
        return np.array([state[feature] for feature in self.STATES])\
                 .reshape(1, len(self.STATES))


class FeaturesLambdaSarsa:
    STATES = [
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
    ALPHA_DIM = NB_FRAMES

    GAMMA = 0.9  # discount factor
    UP_PROBA = 0.1
    EPS0 = 0.4
    ALPHA0 = 0.2  # learning rate
    LAMBDA = 0.8
    SIZE_FIFO = None

    NB_TEST = 100

    DATA_DIREC = 'data/FLS/'

    def __init__(self):
        self.game = FlappyBird()
        self.p = PLE(self.game, fps=30, frame_skip=1, num_steps=1,
                     force_fps=True, display_screen=True)
        self.epsilon = self.EPS0  # epsilon-greddy
        self.alpha = self.ALPHA0
        # (feature1, feature1, feature1): [qval_a1, qval_a2]
        self.Q = {}

    def play(self, n=1):

        self.p = PLE(self.game, fps=30, frame_skip=1, num_steps=1,
                     force_fps=False, display_screen=True)
        for _ in range(n):
            self.p.reset_game()
            while not self.p.game_over():
                nb_not_seen = 0
                state = self.game.getGameState()
                state_tp = self.discretize(state)
                if state_tp not in self.Q:
                    nb_not_seen += 1
                    act = 1 if random.random() < self.UP_PROBA else 0
                else:
                    qval = self.Q[state_tp]
                    act = np.argmax(qval)
                self.p.act(self.ACTIONS[act])
            print('Not seen in this game:', nb_not_seen)

    def test(self):
        self.p = PLE(self.game, fps=30, frame_skip=1, num_steps=1,
                     force_fps=True, display_screen=True)
        cumulated = np.zeros((self.NB_TEST))
        total_not_seen = 0
        for i in range(self.NB_TEST):
            self.p.reset_game()
            nb_not_seen = 0
            while not self.p.game_over():
                state = self.game.getGameState()
                state_tp = self.discretize(state)
                if state_tp not in self.Q:
                    nb_not_seen += 1
                    act = 1 if random.random() < self.UP_PROBA else 0
                else:
                    qval = self.Q[state_tp]
                    act = np.argmax(qval)
                reward = self.p.act(self.ACTIONS[act])
                if reward > 0:
                    cumulated[i] += 1
            print(i, ': Nb not seen:', nb_not_seen)
            total_not_seen += nb_not_seen

        average_score = np.mean(cumulated)
        max_score = np.max(cumulated)
        print()
        print('Test over', self.NB_TEST, 'tests:')
        print('Total not seen:', total_not_seen)
        print('Mean not seen:', total_not_seen/self.NB_TEST)
        print('average_score', 'max_score')
        print(average_score, max_score)
        return average_score, max_score

    def train(self, scratch=True):
        t1 = time.time()
        if scratch:
            delete_files(self.DATA_DIREC)
            f0 = 0
            curr_frame = 0
            nb_save = 0
            nb_games = 0
        else:
            file_name = self.load().split('.')[0]
            nb_save, curr_frame, nb_games = file_name.split('_')[1:]
            nb_save = ord(nb_save) - 97  # !!
            curr_frame, nb_games = int(curr_frame), int(nb_games)
            f0 = curr_frame

        eps_tau = (self.NB_FRAMES - f0)//8

        scores = []
        while curr_frame < self.NB_FRAMES:
            if len(scores) == self.SCORE_FREQ:
                print(''.join([(str(s) if s != 0 else '.') for s in scores]))
                print('States visited:', len(self.Q))
                print('Over the last', self.SCORE_FREQ, 'games:')
                print('    MEAN', sum(scores)/len(scores))
                print('    TOTAL', sum(scores))
                print('############################################')
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
                curr_frame += 1
                if curr_frame != 0 and (curr_frame % self.SAVE_FREQ) == 0:
                    self.save('Q_' + chr(97+nb_save) + '_' + str(curr_frame) +
                              '_' + str(nb_games) + '.p')
                    nb_save += 1

                if curr_frame != 0 and (curr_frame % self.EPS_UPDATE_FREQ) == 0:
                    self.epsilon = self.EPS0*np.exp(-(curr_frame-f0)/eps_tau)
                    print('CURRENT FRAME:', curr_frame,
                          100*curr_frame / self.NB_FRAMES, '%',
                          'EPSILON: ', self.epsilon)
                if curr_frame != 0 and (curr_frame % self.ALPHA_DIM) == 0:
                    print('###---###---###---###---')
                    print('ALPHA halved: ', self.alpha)
                    self.alpha /= 2
                # 1) Observe r, s′
                bare_reward = self.p.act(self.ACTIONS[act])
                reward = self.reward_engineering(bare_reward)
                new_state = self.game.getGameState()
                new_state_tp = self.discretize(new_state)

                # 2) Choose a′ (GLIE actor) using Q
                if new_state_tp not in self.Q:
                    self.Q[new_state_tp] = [0, 0]
                qval = self.Q[new_state_tp]
                if random.random() < self.epsilon:  # exploration
                    new_act = 1 if random.random() < self.UP_PROBA else 0
                else:
                    new_act = np.argmax(qval)

                # 3) Temporal difference:  δ=r+γQ(s′,a′)−Q(s,a)
                delta = reward + self.GAMMA*self.Q[new_state_tp][new_act] - self.Q[state_tp][act]

                # 4) Update Q
                episode.append((state_tp, act))
                elig[(state_tp, act)] = 1
                for (state_tp_ep, act_ep) in episode:
                    self.Q[state_tp_ep][act_ep] += (
                            self.alpha * delta * elig[(state_tp_ep, act_ep)])
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
        self.save('Q_' + chr(97+nb_save) + '_' + str(curr_frame) +
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
        return tuple(state[feature] for feature in self.STATES)

    def reward_engineering(self, reward):
        # if reward >= 0:
        #     reward = 1
        # elif reward < 0:
        #     reward = -1000
        return reward

    def save(self, name):
        with open(os.path.join(self.DATA_DIREC, name), 'bw') as f:
            pickle.dump(self.Q, f)

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


if __name__ == '__main__':
    # athlete = FeaturesNeuralQLearning()
    athlete = FeaturesLambdaSarsa()

    # athlete.train(scratch=False)
    athlete.load()

    average_score, max_score = athlete.test()
    athlete.play(10)
