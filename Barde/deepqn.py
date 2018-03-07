#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 16:56:33 2018

@author: paul
"""
import numpy as np
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten
from keras.utils import plot_model
from keras.optimizers import RMSprop, Adam
from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize
import random
import matplotlib.pyplot as plt

total_steps = 200000  # number of learning steps
mini_batch_size = 32
gamma = 0.95
memory_size = 200000  # size of the memory replay
epsilon_init = 0.11  # initial epsilon for linear and exponential decay
epsilon_final = 0.01  # final epsilon for linear decay
prop_decay = 1  # proportion of total steps during which linear decay happens
epsilon_decay = (epsilon_init - epsilon_final) / (prop_decay * total_steps)
learning_rate = 1e-5
epsilon_tau = 50000  # time constant of exponential decay
nb_games = 10


def epsilon_exp(step):
    """
    Exponential decay of the epsilon greedy policy
    :param step:
    :return:
    """
    return epsilon_init * np.exp(-step / epsilon_tau)


def epsilon_action(step):
    return 0.15 + 0.35 * (1 - np.exp(-0.75 * step / epsilon_tau))


def epsilon(step):
    if step < total_steps * prop_decay:
        return epsilon_init - step * epsilon_decay
    return epsilon_final


def clip_reward(r):
    if r == 0:
        r = 0.1
    elif r < 0:
        r = -1
    return r


def process_screen(x):
    return (255 * resize(rgb2gray(x)[50:, :410], (84, 84))).astype("uint8")


class Agent:
    def __init__(self, model=None):
        self.model = model
        self.memory = MemoryBuffer(memory_size)
        self.buffer_state = deque(maxlen=4)
        self.state = None

    def create_model(self, plot=False, name=''):
        initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

        dqn = Sequential()
        # 1st layer
        dqn.add(Conv2D(filters=16, kernel_size=(8, 8), strides=4, activation="relu", input_shape=(84, 84, 4),
                       kernel_initializer=initializer))
        # 2nd layer
        dqn.add(Conv2D(filters=32, kernel_size=(4, 4), strides=2, activation="relu", kernel_initializer=initializer))
        dqn.add(Flatten())
        # 3rd layer
        dqn.add(Dense(units=256, activation="relu", kernel_initializer=initializer))
        # output layer
        dqn.add(Dense(units=2, activation="linear", kernel_initializer=initializer))

        dqn.compile(optimizer=Adam(lr=learning_rate), loss="mean_squared_error")

        if plot:
            plot_model(dqn, to_file=name + '.png', show_shapes=True)

        self.model = dqn

    def save_model(self, filepath):
        self.model.save(filepath + ".h5")

    def save_weights(self, filepath):
        self.model.save_weights(filepath + ".h5")

    def load_model(self, filepath):
        self.model = load_model(filepath + ".h5")

    def load_weights(self, filepath):
        self.model.load_weights(filepath + ".h5")

    def greedy_action(self, verbose=False):
        Q = self.model.predict(np.array([self.state]))
        if verbose:
            print("Q = {}".format(Q))
            print("greedy = {}".format(np.argmax(Q)))
        return np.argmax(Q)

    def store(self, screen, a, r):
        self.memory.append(screen, a, r)

    def reset_state(self, screen):
        for i in range(4):
            self.buffer_state.append(screen)
        self.state = np.stack(self.buffer_state, axis=-1)

    def update_state(self, screen):
        self.buffer_state.append(screen)
        self.state = np.stack(self.buffer_state, axis=-1)

    def e_greedy_policy(self, step):
        if np.random.rand() < epsilon(step):
            if np.random.rand() < 0.5:
                a_idx = 0
            else:
                a_idx = 1
        else:
            a_idx = self.greedy_action(verbose=False)
        return a_idx

    def learn(self, show=False):
        X, A, R, Y, D = self.memory.minibatch(mini_batch_size, show=show)
        QY = self.model.predict(Y)
        QYmax = QY.max(1)
        # print("QY = {}".format(QY))
        # print("QYmax = {}".format(QYmax))
        update = R + gamma * (1 - D) * QYmax
        # print("R = {}".format(R))
        # print("D = {}".format(D))
        # print("gamma * (1 - D) * QYmax = {}".format(gamma * (1 - D) * QYmax))
        QX = self.model.predict(X)
        # print("QX = {}".format(QX))
        QX[np.arange(mini_batch_size), A.ravel()] = update.ravel()
        # print("QX' = {}".format(QX))
        # print("A = {}".format(A))
        loss = self.model.train_on_batch(x=X, y=QX)
        return loss

    def evaluate_perfs(self, p, ACTIONS):
        cumulated = np.zeros((nb_games))

        for i in range(nb_games):
            p.reset_game()
            while not p.game_over():
                screen = process_screen(p.getScreenRGB())
                self.update_state(screen)
                action = self.greedy_action()  ### Your job is to define this function.
                reward = p.act(ACTIONS[action])
                cumulated[i] = cumulated[i] + reward

        average_score = np.mean(cumulated)
        max_score = np.max(cumulated)

        return average_score, max_score

    def display_state(self):
        for j in range(4):
            ax = plt.subplot(1, 4, j + 1)
            ax.imshow(self.state[:, :, j], cmap="gray")
            ax.set_title("x frame {}".format(j))
        plt.show()


class MemoryBuffer:
    "An experience replay buffer using numpy arrays"

    def __init__(self, length, screen_shape=(84, 84)):
        self.length = length
        self.screen_shape = screen_shape
        self.screens = deque(maxlen=self.length)  # screens
        self.actions = deque(maxlen=self.length)  # actions
        self.rewards = deque(maxlen=self.length)  # rewards

    def append(self, screen, a, r):
        """
        Action a is taken from state which last screen is screen and one gets reward r
        :param screen:
        :param a:
        :param r:
        :return:
        """
        self.screens.append(screen)
        self.actions.append(np.uint8(a))
        self.rewards.append(r)

    def stack_frames(self, index):
        pos = index
        to_be_stacked = []
        rewards = []
        for i in range(4):
            to_be_stacked.insert(0, self.screens[pos])
            rewards.insert(0, self.rewards[pos])
            # if not self.rewards[pos - 1] < 0:
            #     pos = pos - 1
        return np.stack(to_be_stacked, axis=-1), rewards

    def minibatch(self, size, show=False):
        indices = random.sample(range(4, len(self.actions) - 1), size)
        x = np.zeros((size,) + self.screen_shape + (4,))
        y = np.zeros((size,) + self.screen_shape + (4,))
        d = np.zeros((size,))
        A = np.zeros((size,), dtype=np.uint8)
        R = np.zeros((size,))
        for i in range(size):
            x[i], rewards_x = self.stack_frames(indices[i])
            y[i], rewards_y = self.stack_frames(indices[i] + 1)
            A[i] = self.actions[indices[i]]
            R[i] = self.rewards[indices[i]]

            if self.rewards[indices[i]] < 0:
                d[i] = 1

            if show:
                MemoryBuffer.display_minibatch(x[i], y[i], rewards_x, rewards_y, A[i], R[i], d[i])

        return x, A, R, y, d

    @staticmethod
    def display_minibatch(x, y, rewards_x, rewards_y, A, R, d):
        fig = plt.figure()
        for j in range(4):
            ax = plt.subplot(2, 4, j + 1)
            ax.imshow(x[:, :, j], cmap="gray")
            ax.set_title("x frame {}, r local : {}".format(j, rewards_x[j]))
        for j in range(4):
            ax = plt.subplot(2, 4, j + 5)
            ax.imshow(y[:, :, j], cmap="gray")
            ax.set_title("y frame {}, r local : {}".format(j, rewards_y[j]))
        fig.suptitle("reward : {}, d : {}, action : {}".format(R, d, A))
        plt.show()
        # input("Press Enter to continue...")


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""MAIN"""""""""""""""""""

if __name__ == "__main__":
    dqn = Agent()
    dqn.create_model(plot=True, name='mybasicnetwork')
    dqn.save_model('my_basic_network_')
