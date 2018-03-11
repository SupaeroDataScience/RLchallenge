import random
import numpy as np
import os.path
from collections import deque
from datetime import datetime
import pickle

import tensorflow as tf
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout, Conv2D, Flatten
from keras.callbacks import TensorBoard

from sklearn.preprocessing import StandardScaler

from skimage.color import rgb2gray
from skimage.transform import resize


class FlappyPolicy(object):

    def __init__(self, model='dumb', nb_games=1):
        if model == 'dumb':
            self.lm = Dumb_model()
        elif model == 'SARSA':
            self.lm = SARSA(nb_games)
        elif model == 'L_SARSA':
            self.lm = L_SARSA(nb_games)
        elif model == 'L_SARSA_w_AVFA':
            self.lm = L_SARSA_w_AVFA(nb_games)
        elif model == 'Q_LEARNING':
            self.lm = Q_learning(nb_games)
        elif model == 'Q_LEARNING_w_AVFA':
            self.lm = Q_learning_w_AVFA(nb_games)
        elif model == 'DQN':
            self.lm = DQN(nb_games)

    def reset_game(self):
        return self.lm.reset_game()

    def next_action(self, state, screen, r, step):
        return self.lm.next_action(state, screen, r, step)

    def print_settings(self):
        return self.lm.print_settings()

    def save(self):
        return self.lm.save()

    def log(self, names, logs, batch_no):
        self.lm.write_log(self.lm.tbCallBack, names, logs, batch_no)
        print("Logging game performance.")


class DQN:

    def __init__(self, nb_games):

        self.actions = [None, 119]

        self.actions_shape = (2, 1)

        self.state_shape = (1, 8)

        self.update = False

        # Game counters
        self.episode = 1
        self.step = 0
        self.nb_games = nb_games

        # Hyperparameters
        self.epsilon = 1.0  # greedy search parameter #linear TODO
        self.F_EPSILON = 400  # frequence of epsilon updates in steps
        self.K_EPSILON = 1e-3  # coefficient which multiplies epsilon
        self.alpha = 1e-4  # learning rate
        self.gamma = 0.99  # discounted reward factor
        self.DQN_TARGET = 200  # update frequency of second network with target in steps

        # Experience replay
        replay_filename = "Replay_memory"
        if os.path.exists(replay_filename):
            with open(replay_filename, "rb") as f:
                self.replay = pickle.load(f)
                self.update = 1
                print('Previous replay memory successfully imported')
        else:
            print('New replay memory successfully created')

            self.replay = self.Experience_replay(length=100_000,
                                                 screen_shape=(80,80),
                                                 action_shape=(1,))

        # Replay Minibatch size
        self.minibatch_size = 32
        self.WAIT_BEFORE_TRAINING = self.minibatch_size # in steps
        # Minibatch size
        self.batch_size = 32

        # Q estimator
        self.Q_estimator = self.define_estimator()

        # Weights for Double Q-learning
        self.weights = self.Q_estimator.get_weights()
        self.weights_old = self.weights

        # Scaler
        mins = [0, -20, 0, 0, 0, 0, 0, 0]
        maxs = [512, 20, 288, 512, 512, 288, 512, 512]
        self.scaler = StandardScaler().fit([mins, maxs])

        # Frequency of TensorBoard logging in steps
        self.TB_LOG_FREQ = 1_000
        self.total_step = 0

    def reset_game(self):
        self.episode += 1
        self.step = 0

    # def write_log(self,callback, names, logs, batch_no):
    #     for name, value in zip(names, logs):
    #         summary = tf.Summary()
    #         summary_value = summary.value.add()
    #         summary_value.simple_value = value
    #         summary_value.tag = name
    #         callback.writer.add_summary(summary)
    #     callback.writer.flush()

    def write_log(self, callback, names, logs, batch_no):

        summary = tf.Summary()

        for name, value in zip(names, logs):
            #summary_value = tf.summary.scalar(name,value)
            summary = tf.Summary(
                value=[tf.Summary.Value(tag=name, simple_value=value)])

            callback.writer.add_summary(summary)
        # callback.writer.flush()

    def process_screen(self,screen):
        return 255*resize(rgb2gray(screen[60:, 25:310,:]),(80,80))

    def define_estimator(self):

        estimator_filename = "Q_estimator"


        self.tbCallBack = TensorBoard(log_dir=f"./logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                                      histogram_freq=1,
                                      batch_size=self.batch_size,
                                      write_graph=False,
                                      write_grads=True,
                                      write_images=False,
                                      embeddings_freq=0,
                                      embeddings_layer_names=None,
                                      embeddings_metadata=None)

        if os.path.exists(estimator_filename):
            model = load_model(estimator_filename)
            print('Previous Q_estimator successfully imported')

        else:

            dropout_rate = 0.1

            model = Sequential()
            # self.tbCallBack.set_model(model)

            # First relu layer
            model.add(Conv2D(filters=16, kernel_size=(8,8), strides=4, activation="relu", input_shape=(80,80,4)))
            model.add(Conv2D(filters=32, kernel_size=(4,4), strides=2,activation="relu"))
            model.add(Flatten())

            model.add(Dense(units=128,
                            kernel_initializer='lecun_uniform',
                            bias_initializer='lecun_uniform'))
            model.add(Activation('relu'))
            #model.add(Dropout(dropout_rate))

            # Second relu layer
            # model.add(Dense(units=64,
            #                 kernel_initializer='random_uniform',
            #                 bias_initializer='random_uniform'))
            # model.add(Activation('relu'))
            #model.add(Dropout(dropout_rate))

            # Final softmax later
            model.add(Dense(units=2,
                            kernel_initializer='lecun_uniform',
                            bias_initializer='lecun_uniform'))
            model.add(Activation('linear'))

            # Model parameters
            model.compile(loss="mean_squared_error",
                          optimizer=keras.optimizers.Adam(lr=self.alpha),
                          metrics=['mse'])

        model.summary()

        return model

    def getQ(self, X, old_weights=False, action=None):
        # print(f"getQ of {state} and {action}")
        # if old_weights:
            # self.Q_estimator.set_weights(self.weights_old)
        if action is not None:
            # X = np.hstack((state, action)).reshape(
            #     (-1, state.shape[1]+1))

            Q = self.Q_estimator.predict(X,
                                         batch_size=self.minibatch_size)[range(X.shape[0]), action]
        else:
            # print(state.shape)
            # X1 = np.hstack(
            #     (state, np.zeros((state.shape[0], 1)))).reshape((-1, state.shape[1]+1))
            # X2 = np.hstack(
            #     (state, np.ones((state.shape[0], 1)))).reshape((-1, state.shape[1]+1))
            # X1 = self.scaler.transform(X1)
            # X2 = self.scaler.transform(X2)
            # Q = [self.Q_estimator.predict(X1,
            #                                  batch_size=self.minibatch_size),
            #         self.Q_estimator.predict(X2,
            #                                  batch_size=self.minibatch_size)]

            Q = self.Q_estimator.predict(X,
                                         batch_size=self.minibatch_size)

        # if old_weights:
            # self.Q_estimator.set_weights(self.weights)

        return Q

    def updateQ(self, replay):
        # print(replay.minibatch(self.minibatch_size))

        X, a, r, X2, t = replay.minibatch(self.minibatch_size)

        # Format state
        Y = self.getQ(X)

        # Simple QN
        update = r + self.gamma * self.getQ(X2).max(axis=1)
        Y[range(Y.shape[0]), a] = update

        # Double QN (van Hasselt)
        #Y[range(Y.shape[0]), a] = r + self.gamma * self.getQ(s2, action=np.argmax(
        #     self.getQ(s2), axis=1), old_weights=True)

        # Update prioritized experience replay
        #replay.update_p_minibatch(np.abs(Y - self.getQ(s, action=a, weights=self.weights)) + 0.001)

        # If terminal state Y = r
        Y[range(Y.shape[0]), a][np.where(r.reshape(-1, 1) < 0)] = -1.0


        # Log to Tensorboard each TB_LOG_FREQ steps
        if((self.total_step) % self.TB_LOG_FREQ == 0):
            self.Q_estimator.fit(X, Y,
                                 epochs=1,
                                 validation_split=0.1,
                                 batch_size=self.minibatch_size,
                                 verbose=0,
                                 callbacks=[self.tbCallBack])
        else:
            self.Q_estimator.train_on_batch(X, Y)
            # self.Q_estimator.fit(X, Y,
            #                          epochs=1,
            #                          validation_split=0.1,
            #                          batch_size=self.batch_size,
            #                          verbose=0)
        # if((self.total_step+1) % self.DQN_TARGET == 0):
        #     self.weights_old = self.weights
        #     self.weights = self.Q_estimator.get_weights()

    def start_episode(self, state, screen):

        self.s = self.process_screen(screen)

        screen_x = self.s
        self.stacked_x = deque([screen_x, screen_x, screen_x, screen_x], maxlen=4)
        X = np.expand_dims(np.stack(self.stacked_x, axis=-1),axis=0)

        self.a = self.epsilon_greedy(self.getQ(X), self.epsilon)

        return self.actions[self.a]

    def next_action(self, state, screen, r, total_step):
        # Initialization
        self.total_step = total_step

        if self.step == 0:
            self.step += 1
            return self.start_episode(state, screen)

        # Then each step of the episode do:
        else:
            s2 = self.process_screen(screen)

            # Update the experience replay memory
            self.replay.append(self.s, self.a, r, s2, 0)

            # If experience replay is full update Q
            if not self.update and total_step >= self.WAIT_BEFORE_TRAINING:
                self.update = True
                print("\n Starting training \n")

            elif self.update:
                if((self.total_step+1) % self.F_EPSILON == 0):
                    self.epsilon -= self.K_EPSILON
                    self.epsilon = np.max([0.1, self.epsilon])

                # Batch update Q estimator
                self.updateQ(self.replay)

            self.stacked_x.append(s2)
            X = np.expand_dims(np.stack(self.stacked_x, axis=-1),axis=0)

            Qs2 = self.getQ(X)

            # Compute next action
            a2 = self.epsilon_greedy(Qs2, self.epsilon)
            self.a = a2

            self.step += 1

            return self.actions[a2]

    def epsilon_greedy(self, Qs, epsilon):
        if(np.random.rand() <= epsilon):  # random action
            a = np.random.choice([0, 1], p=[0.9, 0.1])
        else:
            a = Qs.argmax()

        # print('action_taken',a)
        return a

    def save(self):
        self.Q_estimator.save("Q_estimator")
        print('\n\n Q estimator correctly saved.')

        self.replay.save("Replay_memory")
        print(' Replay memory correctly saved.')

    def print_settings(self):
        # Hyperparameters
        print(f' | epsilon: \t \t {self.epsilon:.4f}')
        print(f' | alpha: \t \t \t {self.alpha:.0f}')
        print(f' | gamma: \t \t \t {self.gamma:.4f}')
        print(f' | # of steps: \t \t {self.total_step:_}')
    

    class Experience_replay():
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
            self.actions[self.index] = a
            self.rewards[self.index] = r
            self.screens_y[self.index] = screeny
            self.terminals[self.index] = d
            self.index = (self.index+1) % self.length
            self.size = np.min([self.size+1,self.length])

        def stacked_frames_x(self, index):
            im_deque = deque(maxlen=4)
            pos = index % self.length
            for i in range(4):
                im = self.screens_x[pos]
                im_deque.appendleft(im)
                test_pos = (pos-1) % self.length
                if self.terminals[test_pos] == False:
                    pos = test_pos
            return np.stack(im_deque, axis=-1)

        def stacked_frames_y(self, index):
            im_deque = deque(maxlen=4)
            pos = index % self.length
            for i in range(4):
                im = self.screens_y[pos]
                im_deque.appendleft(im)
                test_pos = (pos-1) % self.length
                if self.terminals[test_pos] == False:
                    pos = test_pos
            return np.stack(im_deque, axis=-1)

        def minibatch(self, size):
            indices = np.random.choice(self.size, size=size, replace=False)
            x = np.zeros((size,)+self.screen_shape+(4,))
            y = np.zeros((size,)+self.screen_shape+(4,))
            for i in range(size):
                x[i] = self.stacked_frames_x(indices[i])
                y[i] = self.stacked_frames_y(indices[i])
            return x, self.actions[indices], self.rewards[indices], y, self.terminals[indices]

    """
    class Experience_replay():
        def __init__(self, size, prioritization, state_shape, actions_shape):

            self.state_shape = state_shape
            self.actions_shape = actions_shape

            from memory_tree import SumTree
            self.replay_size = size
            self.tree = SumTree(size)

            self.alpha = prioritization

        def append(self, s, a, r, s2):
            self.tree.add(1, (s, a, r, s2))

        def minibatch(self, size, episode):

            s = np.empty((size, self.state_shape[1]))
            a = np.empty((size, self.actions_shape[1]))
            r = np.empty((size, 1))
            s2 = np.empty((size, self.state_shape[1]))

            self.indexes = np.empty(size)

            for i, p in enumerate(1+np.random.choice(int(self.tree.total())-1, size=size)):
                (idx, tree_idx, data) = self.tree.get(p)
                (s_t, a_t, r_t, s2_t) = data
                s[i, :] = s_t
                a[i, :] = a_t
                r[i, :] = r_t
                s2[i, :] = s2_t
                self.indexes[i] = idx

            return s, a, r, s2

        def update_p_minibatch(self, delta):
            for i, index in enumerate(self.indexes):
                self.tree.update(int(index), delta[i][0])

        def save(self, filename):
            with open(filename, "wb") as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
    """


class Q_learning_w_AVFA:

    def __init__(self, nb_games):

        self.actions = [None, 119]

        self.actions_shape = (2, 1)

        self.state_shape = (1, 8)

        self.update = False

        # Game counters
        self.episode = 1
        self.step = 0
        self.nb_games = nb_games

        # Hyperparameters
        self.epsilon = 1.0  # greedy search parameter #linear TODO
        self.F_EPSILON = 400  # frequence of epsilon updates in steps
        self.K_EPSILON = 1e-3  # coefficient which multiplies epsilon
        self.alpha = 1e-4  # learning rate
        self.gamma = 0.99  # discounted reward factor
        self.DQN_TARGET = 200  # update frequency of second network with target in steps

        # Experience replay
        replay_filename = "Replay_memory"
        if os.path.exists(replay_filename):
            with open(replay_filename, "rb") as f:
                self.replay = pickle.load(f)
                self.update = 1
                print('Previous replay memory successfully imported')
        else:
            print('New replay memory successfully created')
            self.replay = self.Experience_replay(size=100_000,
                                                 state_shape=self.state_shape,
                                                 actions_shape=self.actions_shape)

        # Replay Minibatch size
        self.minibatch_size = 32
        self.WAIT_BEFORE_TRAINING = self.minibatch_size # in steps
        # Minibatch size
        self.batch_size = 32

        # Q estimator
        self.Q_estimator = self.define_estimator()

        # Weights for Double Q-learning
        self.weights = self.Q_estimator.get_weights()
        self.weights_old = self.weights

        # Scaler
        mins = [0, -20, 0, 0, 0, 0, 0, 0]
        maxs = [512, 20, 288, 512, 512, 288, 512, 512]
        self.scaler = StandardScaler().fit([mins, maxs])

        # Frequency of TensorBoard logging in steps
        self.TB_LOG_FREQ = 1_000
        self.total_step = 0

    def reset_game(self):
        self.episode += 1
        self.step = 0

    # def write_log(self,callback, names, logs, batch_no):
    #     for name, value in zip(names, logs):
    #         summary = tf.Summary()
    #         summary_value = summary.value.add()
    #         summary_value.simple_value = value
    #         summary_value.tag = name
    #         callback.writer.add_summary(summary)
    #     callback.writer.flush()

    def write_log(self, callback, names, logs, batch_no):

        summary = tf.Summary()

        for name, value in zip(names, logs):
            #summary_value = tf.summary.scalar(name,value)
            summary = tf.Summary(
                value=[tf.Summary.Value(tag=name, simple_value=value)])

            callback.writer.add_summary(summary)
        # callback.writer.flush()

    def define_estimator(self):

        estimator_filename = "Q_estimator"


        self.tbCallBack = TensorBoard(log_dir=f"./logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                                      histogram_freq=1,
                                      batch_size=self.batch_size,
                                      write_graph=False,
                                      write_grads=True,
                                      write_images=False,
                                      embeddings_freq=0,
                                      embeddings_layer_names=None,
                                      embeddings_metadata=None)

        if os.path.exists(estimator_filename):
            model = load_model(estimator_filename)
            print('Previous Q_estimator successfully imported')

        else:

            dropout_rate = 0.1

            model = Sequential()
            # self.tbCallBack.set_model(model)

            # First relu layer
            model.add(Dense(units=64,
                            input_dim=self.state_shape[1],
                            kernel_initializer='lecun_uniform',
                            bias_initializer='lecun_uniform'))
            model.add(Activation('relu'))
            #model.add(Dropout(dropout_rate))

            # Second relu layer
            # model.add(Dense(units=64,
            #                 kernel_initializer='random_uniform',
            #                 bias_initializer='random_uniform'))
            # model.add(Activation('relu'))
            #model.add(Dropout(dropout_rate))

            # Final softmax later
            model.add(Dense(units=2,
                            kernel_initializer='lecun_uniform',
                            bias_initializer='lecun_uniform'))
            model.add(Activation('linear'))

            # Model parameters
            model.compile(loss="mean_squared_error",
                          optimizer=keras.optimizers.Adam(lr=self.alpha),
                          metrics=['mse'])

        model.summary()

        return model

    def getQ(self, state, old_weights=False, action=None):
        # print(f"getQ of {state} and {action}")
        # if old_weights:
            # self.Q_estimator.set_weights(self.weights_old)
        if action is not None:
            # X = np.hstack((state, action)).reshape(
            #     (-1, state.shape[1]+1))
            X = state.reshape(-1, state.shape[1])

            assert X.shape[0] == action.shape[0]

            X = self.scaler.transform(X)
            Q = self.Q_estimator.predict(X,
                                         batch_size=self.minibatch_size)[range(X.shape[0]), action]
        else:
            # print(state.shape)
            # X1 = np.hstack(
            #     (state, np.zeros((state.shape[0], 1)))).reshape((-1, state.shape[1]+1))
            # X2 = np.hstack(
            #     (state, np.ones((state.shape[0], 1)))).reshape((-1, state.shape[1]+1))
            # X1 = self.scaler.transform(X1)
            # X2 = self.scaler.transform(X2)
            # Q = [self.Q_estimator.predict(X1,
            #                                  batch_size=self.minibatch_size),
            #         self.Q_estimator.predict(X2,
            #                                  batch_size=self.minibatch_size)]
            X = state.reshape(-1, state.shape[1])
            X = self.scaler.transform(X)
            Q = self.Q_estimator.predict(X,
                                         batch_size=self.minibatch_size)

        # if old_weights:
            # self.Q_estimator.set_weights(self.weights)

        return Q

    def updateQ(self, replay):
        # print(replay.minibatch(self.minibatch_size))

        s, a, r, s2 = replay.minibatch(
            self.minibatch_size, episode=self.episode)

        # Format state
        X = s
        Y = self.getQ(s)

        # Simple QN
        Y[range(Y.shape[0]), a] = r + self.gamma * \
            self.getQ(s2).max(axis=1)

        # Double QN (van Hasselt)
        # Y[range(Y.shape[0]), a] = r + self.gamma * self.getQ(s2, action=np.argmax(
        #     self.getQ(s2), axis=1), old_weights=True)

        # Update prioritized experience replay
        #replay.update_p_minibatch(np.abs(Y - self.getQ(s, action=a, weights=self.weights)) + 0.001)

        # If terminal state Y = r
        Y[range(Y.shape[0]), a][np.where(r.reshape(-1, 1) < 0)] = -1.0

        X = self.scaler.transform(X)

        # Log to Tensorboard each TB_LOG_FREQ steps
        if((self.total_step) % self.TB_LOG_FREQ == 0):
            self.Q_estimator.fit(X, Y,
                                 epochs=1,
                                 validation_split=0.1,
                                 batch_size=self.minibatch_size,
                                 verbose=0,
                                 callbacks=[self.tbCallBack])
        else:
            self.Q_estimator.train_on_batch(X, Y)
            # self.Q_estimator.fit(X, Y,
            #                          epochs=1,
            #                          validation_split=0.1,
            #                          batch_size=self.batch_size,
            #                          verbose=0)
        # if((self.total_step+1) % self.DQN_TARGET == 0):
        #     self.weights_old = self.weights
        #     self.weights = self.Q_estimator.get_weights()

    def start_episode(self, state, screen):

        self.s = np.array([list(state.values())])
        self.a = self.epsilon_greedy(
            self.getQ(self.s), self.epsilon)

        return self.actions[self.a]

    def next_action(self, state, screen, r, total_step):
        # Initialization
        self.total_step = total_step

        if self.step == 0:
            self.step += 1
            return self.start_episode(state, screen)

        # Then each step of the episode do:
        else:
            s2 = np.array([list(state.values())])

            # Update the experience replay memory
            self.replay.append(self.s, self.a, r, s2)

            # If experience replay is full update Q
            if not self.update and total_step >= self.WAIT_BEFORE_TRAINING:
                self.update = True
                print("\n Starting training \n")

            elif self.update:
                if((self.total_step+1) % self.F_EPSILON == 0):
                    self.epsilon -= self.K_EPSILON
                    self.epsilon = np.max([0.1, self.epsilon])

                # Batch update Q estimator
                self.updateQ(self.replay)

            Qs2 = self.getQ(s2)

            # Compute next action
            a2 = self.epsilon_greedy(Qs2, self.epsilon)
            if self.total_step > 300_000:
                print(f"{Qs2} -> {a2}")

            self.a = a2

            self.step += 1

            return self.actions[a2]

    def epsilon_greedy(self, Qs, epsilon):
        if(np.random.rand() <= epsilon):  # random action
            a = np.random.choice([0, 1], p=[0.9, 0.1])
        else:
            a = Qs.argmax()

        # print('action_taken',a)
        return a

    def save(self):
        self.Q_estimator.save("Q_estimator")
        print('\n\n Q estimator correctly saved.')

        self.replay.save("Replay_memory")
        print(' Replay memory correctly saved.')

    def print_settings(self):
        # Hyperparameters
        print(f' | epsilon: \t \t {self.epsilon:.4f}')
        print(f' | alpha: \t \t \t {self.alpha:.0f}')
        print(f' | gamma: \t \t \t {self.gamma:.4f}')
        print(f' | +1 rewards in replay: \t {len(self.replay.r[self.replay.r>0])}')
        print(f' | # of steps: \t \t {self.total_step:_}')

    """
    class Experience_replay():
        def __init__(self, size, prioritization, state_shape, actions_shape):

            self.state_shape = state_shape
            self.actions_shape = actions_shape

            from memory_tree import SumTree
            self.replay_size = size
            self.tree = SumTree(size)

            self.alpha = prioritization

        def append(self, s, a, r, s2):
            self.tree.add(1, (s, a, r, s2))

        def minibatch(self, size, episode):

            s = np.empty((size, self.state_shape[1]))
            a = np.empty((size, self.actions_shape[1]))
            r = np.empty((size, 1))
            s2 = np.empty((size, self.state_shape[1]))

            self.indexes = np.empty(size)

            for i, p in enumerate(1+np.random.choice(int(self.tree.total())-1, size=size)):
                (idx, tree_idx, data) = self.tree.get(p)
                (s_t, a_t, r_t, s2_t) = data
                s[i, :] = s_t
                a[i, :] = a_t
                r[i, :] = r_t
                s2[i, :] = s2_t
                self.indexes[i] = idx

            return s, a, r, s2

        def update_p_minibatch(self, delta):
            for i, index in enumerate(self.indexes):
                self.tree.update(int(index), delta[i][0])

        def save(self, filename):
            with open(filename, "wb") as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
    """

    class Experience_replay():
        def __init__(self, size, state_shape, actions_shape):
            self.replay_size = size
            self.s = np.zeros((size, state_shape[1]), dtype='int8')
            self.a = np.zeros((size, actions_shape[1]), dtype='int8')
            self.r = np.zeros((size, 1), dtype='int8')
            self.s2 = np.zeros((size, state_shape[1]), dtype='int8')

            self.state_shape = state_shape
            self.actions_shape = actions_shape

            self.i = 0

        def append(self, s, a, r, s2):
            self.s[self.i] = s
            self.a[self.i] = a
            self.r[self.i] = r
            self.s2[self.i] = s2

            if self.i <= self.replay_size - 2:
                self.i += 1
            else:
                self.i = 0

        def minibatch(self, size, episode):

            # self.indexes = np.random.choice(len(self.a),
            #                                 size=size,
            #                                 replace=False)
            self.indexes = np.random.randint(0, len(self.a),
                                             size=size)

            s = self.s[self.indexes]#.reshape(-1, self.state_shape[1])
            a = self.a[self.indexes]#.reshape(-1, self.actions_shape[1])
            r = self.r[self.indexes]#.reshape(-1, 1)
            s2 = self.s2[self.indexes]#.reshape(-1, self.state_shape[1])

            return s, a, r, s2

        def update_p_minibatch(self, delta):
            pass

        def save(self, filename):
            with open(filename, "wb") as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)


class Q_learning:

    def __init__(self, nb_games):

        self.actions = [None, 119]

        self.actions_shape = len(self.actions)

        self.state_min = np.array([0,
                                   -48,
                                   -10])

        self.state_shape = np.array([round((284)/4+1),
                                     round(704/8+1),
                                     round(40/2+1)])

        self.state_len = np.prod(self.state_shape)

        self.bin_c = np.array([
            4,
            8,
            2])

        self.episode = 1
        self.step = 0
        self.nb_games = nb_games

        # Hyperparameters
        self.epsilon = 0  # 0.1  # greedy search parameter
        self.alpha = 0  # 0.05 # 0.005 # learning rate
        self.gamma = 0.9  # discounted reward factor

        if os.path.exists('Qmatrix.npy'):
            self.Q = np.load('Qmatrix.npy')
            print('Previous Q matrix successfully imported')
        else:
            #self.Q = np.random.rand(np.prod(self.state_shape), self.actions_shape)
            self.Q = np.zeros((np.prod(self.state_shape), self.actions_shape))

    def get_engineered_state(self, state):
        y = state['player_y']
        speed = state['player_vel']
        next_y_b = state['next_pipe_bottom_y']
        next_dist = state['next_pipe_dist_to_player']

        engineered_state = np.array([next_dist,
                                     next_y_b - y,
                                     speed])

        #print('primal engineered state',engineered_state)
        s = np.round(engineered_state/self.bin_c - self.state_min).astype(int)
        # TODO
        #s = np.digitize(engineered_state,self.state_bins)

        # print(f"Y distance to bottom: {engineered_state[1]} -> {s[1]}")
        return s

    def state_index(self, s):
        try:
            index = np.ravel_multi_index(s, self.state_shape)
        except Exception as e:
            print('state', s)
            print('desired shape', self.state_shape)
            raise e
        return index

    def reset_game(self):
        self.episode += 1
        self.step = 0

    def start_episode(self, state, screen):
        s = self.get_engineered_state(state)
        self.previous_pipe_x_dist = s[0]
        self.s = self.state_index(s)
        self.a = self.epsilon_greedy(self.Q, self.s, self.epsilon)
        # Reduce epislon each 500 episodes
        if((self.episode+1) % 100 == 0):
            self.epsilon *= 0.9

        return self.actions[self.a]

    def next_action(self, state, screen, r):
        # Initialization
        if self.step == 0:
            self.step += 1
            return self.start_episode(state, screen)

        # Then each step of the episode do:
        else:
            s2 = self.get_engineered_state(state)
            s2 = self.state_index(s2)

            # Update Q - Q-learning update
            delta = self.alpha * \
                (r + self.gamma *
                 np.max(self.Q[s2, :]) - self.Q[self.s, self.a])

            self.Q[self.s, self.a] += delta

            # Compute next action
            a2 = self.epsilon_greedy(self.Q, s2, self.epsilon)

            self.s = s2
            self.a = a2

            self.step += 1

            return self.actions[a2]

    def epsilon_greedy(self, Q, s, epsilon):
        a = np.argmax(Q[s, :])
        if(np.random.rand() <= epsilon):  # random action
            a = np.random.randint(self.actions_shape)
        # print('action_taken',a)
        return a

    def save(self):
        np.save('Qmatrix', self.Q)
        print()
        print('Matrix Q correctly saved.')

    def print_settings(self):
        # Hyperparameters
        print(f' | epsilon: {self.epsilon}')
        print(f' | alpha: {self.alpha}')
        print(f' | gamma: {self.gamma}')


class L_SARSA_w_AVFA:

    def __init__(self, nb_games):

        self.actions = [None, 119]

        self.actions_shape = len(self.actions)

        self.state_len = 8

        # Game parameters initialization
        self.episode = 1
        self.step = 0
        self.nb_games = nb_games

        # Hyperparameters
        self.epsilon = 0.1  # greedy search parameter
        self.alpha = 0.01  # learning rate
        self.gamma = 0.9  # discounted reward factor
        self.LAMBDA = 0.9  # eligibility trace factor

        # State bounds
        self.state_min = np.array([0,
                                   -40,
                                   0,
                                   0,
                                   0,
                                   0,
                                   0,
                                   0])

        self.state_max = np.array([284,
                                   20,
                                   284,
                                   284,
                                   679,
                                   284,
                                   284,
                                   679])

        # Load previous action-value function approximation parameters if any
        # else initialize with zeros
        if os.path.exists('theta.npy'):
            self.theta = np.load('theta.npy')
            print('Previous theta parameters successfully imported')
        else:
            self.theta = np.zeros(self.state_len*self.actions_shape+1)

    def get_vector_state(self, state):

        s = np.array([state['player_y'],
                      state['player_vel'],
                      state['next_pipe_bottom_y'],
                      state['next_pipe_top_y'],
                      state['next_pipe_dist_to_player'],
                      state['next_next_pipe_bottom_y'],
                      state['next_next_pipe_top_y'],
                      state['next_next_pipe_dist_to_player']])

        # Center and scale the state vector
        return (s - self.state_min) / (self.state_max - self.state_min)

    def get_vector_state_action(self, state, action):
        s = np.append(state, state)
        s = np.append(s, [1])
        if action:
            s[0:self.state_len] = 0
        else:
            s[self.state_len:-1] = 0

        return s

    def reset_game(self):
        self.episode += 1
        self.step = 0

    def start_episode(self, state, screen):
        # Get vector state
        s2 = self.get_vector_state(state)

        # Pick next action following policy
        a = self.epsilon_greedy(self.theta, s2, self.epsilon)

        # Get vector state action
        self.sa = self.get_vector_state_action(s2, a)

        # Initialize eligibility trace
        self.e = np.zeros(self.state_len*self.actions_shape+1)

        # Reduce epislon each 500 episodes
        if((self.episode+1) % 500 == 0):
            self.epsilon = self.epsilon*0.9

        return self.actions[a]

    def next_action(self, state, screen, r):
        # Initialization
        if self.step == 0:
            self.step += 1
            return self.start_episode(state, screen)

        # Then each step of the episode do:
        else:
            # Get vector state
            s2 = self.get_vector_state(state)

            # Pick next action following policy
            a = self.epsilon_greedy(self.theta, s2, self.epsilon)

            # Get vector state action
            sa2 = self.get_vector_state_action(s2, a)

            # Update theta - SARSA lambda update (1/2)
            delta = r + self.gamma * \
                self.theta.dot(sa2) - self.theta.dot(self.sa)

            # Eligibility trace accumulation and decay
            self.e = self.sa + \
                self.gamma * self.LAMBDA * self.e

            # Update theta - SARSA lambda update (2/2)
            self.theta += self.alpha * delta * self.e

            # Update previous state action value
            self.sa = sa2

            self.step += 1

            return self.actions[a]

    def print_settings(self):
        # Hyperparameters
        print(f' | epsilon: {self.epsilon}')
        print(f' | alpha: {self.alpha}')
        print(f' | gamma: {self.gamma}')
        print(f' | lambda: {self.LAMBDA}')

    def epsilon_greedy(self, theta, s, epsilon):

        sa0 = self.get_vector_state_action(s, 0)
        sa1 = self.get_vector_state_action(s, 1)
        a = np.argmax([self.theta.dot(sa0), self.theta.dot(sa1)])

        if(np.random.rand() <= epsilon):  # random action
            a = np.random.randint(self.actions_shape)
        # print('action_taken',a)
        return a

    def save(self):
        np.save('theta', self.theta)
        print()
        print('theta coefficients correctly saved.')


class L_SARSA:

    def __init__(self, nb_games):

        self.actions = [None, 119]

        self.actions_shape = len(self.actions)

        self.state_min = np.array([0,
                                   -48,
                                   -10])

        self.state_shape = np.array([round((284)/4+1),
                                     round(704/8+1),
                                     round(40/2+1)])

        self.state_len = np.prod(self.state_shape)

        # self.state_bins = np.array([
        #    4,
        #    8,
        #    2])

        self.episode = 1
        self.step = 0
        self.nb_games = nb_games

        # Hyperparameters
        self.epsilon = 0  # 0.1 # greedy search parameter
        self.alpha = 0  # 0.01 # learning rate
        self.gamma = 0.9  # discounted reward factor
        self.LAMBDA = 0.4  # eligibility trace factor

        if os.path.exists('Qmatrix.npy'):
            self.Q = np.load('Qmatrix.npy')
            print('Previous Q matrix successfully imported')
        else:
            #self.Q = np.random.rand(np.prod(self.state_shape), self.actions_shape)
            self.Q = np.zeros((np.prod(self.state_shape), self.actions_shape))

        self.e = np.zeros((np.prod(self.state_shape), self.actions_shape))

    def get_engineered_state(self, state):
        y = state['player_y']
        speed = state['player_vel']
        next_y_b = state['next_pipe_bottom_y']
        next_dist = state['next_pipe_dist_to_player']

        engineered_state = np.array([next_dist,
                                     next_y_b - y,
                                     speed])

        bin_c = np.array([
            4,
            8,
            2])
        #print('primal engineered state',engineered_state)
        s = np.round(engineered_state/bin_c - self.state_min).astype(int)
        # TODO
        #s = np.digitize(engineered_state,self.state_bins)

        # print(f"Y distance to bottom: {engineered_state[1]} -> {s[1]}")
        return s

    def state_index(self, s):
        try:
            index = np.ravel_multi_index(s, self.state_shape)
        except Exception as e:
            print('state', s)
            print('desired shape', self.state_shape)
            raise e
        return index

    def reset_game(self):
        self.episode += 1
        self.step = 0

    def start_episode(self, state, screen):

        # Reduce epislon each 500 episodes
        if((self.episode+1) % 500 == 0):
            self.epsilon = self.epsilon*0.9

        s = self.get_engineered_state(state)
        self.previous_pipe_x_dist = s[0]
        self.s = self.state_index(s)
        self.a = self.epsilon_greedy(self.Q, self.s, self.epsilon)
        self.e = np.zeros((self.state_len, self.actions_shape))
        return self.actions[self.a]

    def next_action(self, state, screen, r):
        # Initialization
        if self.step == 0:
            self.step += 1
            return self.start_episode(state, screen)

        # Then each step of the episode do:
        else:
            s2 = self.get_engineered_state(state)
            s2 = self.state_index(s2)

            # Compute next action
            a2 = self.epsilon_greedy(self.Q, s2, self.epsilon)

            # Update Q - SARSA update
            delta = r + self.gamma * self.Q[s2, a2] - self.Q[self.s, self.a]
            self.Q[self.s, self.a] += self.alpha * delta

            # Eligibility trace accumulation
            self.e[self.s, self.a] += 1

            # Lambda update
            self.Q += self.alpha * delta * self.e

            # Eligibility trace decay
            self.e *= self.gamma * self.LAMBDA

            self.s = s2
            self.a = a2

            self.step += 1

            return self.actions[a2]

    def epsilon_greedy(self, Q, s, epsilon):
        a = np.argmax(Q[s, :])
        if(np.random.rand() <= epsilon):  # random action
            a = np.random.randint(self.actions_shape)
        # print('action_taken',a)
        return a

    def print_settings(self):
        # Hyperparameters
        print(f' | epsilon: {self.epsilon}')
        print(f' | alpha: {self.alpha}')
        print(f' | gamma: {self.gamma}')
        print(f' | lambda: {self.LAMBDA}')

    def save(self):
        np.save('Qmatrix', self.Q)
        print()
        print('Matrix Q correctly saved.')


class SARSA:

    def __init__(self, nb_games):
        self.epsilon = 0.3

        self.actions = [None, 119]

        self.actions_shape = len(self.actions)

        self.state_min = np.array([0,
                                   -48,
                                   -10])

        self.state_shape = np.array([round((284)/4+1),
                                     round(679/8+1),
                                     round(40/2+1)])

        self.state_len = np.prod(self.state_shape)

        self.episode = 1
        self.step = 0
        self.nb_games = nb_games

        self.alpha = 0.001
        self.gamma = 0.9

        self.previous_pipe_x_dist = 999

        if os.path.exists('Qmatrix.npy'):
            self.Q = np.load('Qmatrix.npy')
            print('Previous Q matrix successfully imported')
        else:
            self.Q = np.zeros((np.prod(self.state_shape), self.actions_shape))

    def get_engineered_state(self, state):
        y = state['player_y']
        speed = state['player_vel']
        next_y_b = state['next_pipe_bottom_y']
        next_dist = state['next_pipe_dist_to_player']

        engineered_state = np.array([next_dist,
                                     next_y_b - y,
                                     speed])

        bin_c = np.array([
            4,
            8,
            2])
        #print('primal engineered state',engineered_state)
        s = np.round(engineered_state/bin_c - self.state_min).astype(int)

        return s

    def state_index(self, s):
        # print('state',s)
        #print('desired shape',self.state_shape)
        return np.ravel_multi_index(s, self.state_shape)

    def reset_game(self):
        # If it was the last episode, we save the Q matrix
        if self.episode == self.nb_games:
            self.end_game()

        else:
            self.episode += 1
            self.step = 0

    def start_episode(self, state, screen):
        s = self.get_engineered_state(state)
        self.previous_pipe_x_dist = s[0]
        self.s = self.state_index(s)
        self.a = self.epsilon_greedy(self.Q, self.s, self.epsilon)
        return self.actions[self.a]

    def next_action(self, state, screen, r):
        # Initialization
        if self.step == 0:
            self.step += 1
            return self.start_episode(state, screen)

        # Then each step of the episode do:
        else:
            s2 = self.get_engineered_state(state)
            s2 = self.state_index(s2)

            # Reduce epislon each 500 steps
            if((self.step+1) % 500 == 0):
                self.epsilon = self.epsilon/0.9

            # Compute next action
            a2 = self.epsilon_greedy(self.Q, s2, self.epsilon)

            # Update Q
            self.Q[self.s, self.a] = self.Q[self.s, self.a] + \
                self.alpha * \
                (r + self.gamma * self.Q[s2, a2] - self.Q[self.s, self.a])

            self.s = s2
            self.a = a2

            self.step += 1

            return self.actions[a2]

    def epsilon_greedy(self, Q, s, epsilon):
        a = np.argmax(Q[s, :])
        if(np.random.rand() <= epsilon):  # random action
            a = np.random.randint(self.actions_shape)
        # print('action_taken',a)
        return a

    def end_game(self):
        np.save('Qmatrix', self.Q)

    def print_settings(self):
        # Hyperparameters
        print(f' | epsilon: {self.epsilon}')
        print(f' | alpha: {self.alpha}')
        print(f' | gamma: {self.gamma}')
        print(f' | lambda: {self.LAMBDA}')


class Dumb_model:

    def reset_game(self):
        1+1

    def next_action(self, state, screen):
        actions_allowed = [119, None]

        y = state['player_y']
        speed = state['player_vel']
        next_y_b = state['next_pipe_bottom_y']
        next_y_t = state['next_pipe_top_y']
        next_width = next_y_b - next_y_t

        if y >= next_y_b - next_width/2  \
                and 1 or (state['next_pipe_dist_to_player'] < 130
                          or state['next_pipe_dist_to_player'] > 200) \
                or y > 250:
            return 119
        else:
            return 0

        action = actions_allowed[random.randint(0, 1)]
        return action
