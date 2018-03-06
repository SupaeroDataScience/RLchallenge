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
from keras.optimizers import RMSprop
from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize
import random
import matplotlib.pyplot as plt

total_steps = 200000  # number of learning steps
mini_batch_size = 32
gamma = 0.95
memory_size = 200000  # size of the memory replay
epsilon_init = 0.1  # initial epsilon for linear and exponential decay
epsilon_final = 0.01  # final epsilon for linear decay
prop_decay = 1  # proportion of total steps during which linear decay happens
epsilon_decay = (epsilon_init - epsilon_final) / (prop_decay * total_steps)
learning_rate = 1e-5
epsilon_tau = 50000  # time constant of exponential decay


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


class DQN:
    def __init__(self, model=None):
        self.model = model

    def create_model(self, plot=False, name=''):
        initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)

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

        dqn.compile(optimizer=RMSprop(lr=learning_rate), loss="mean_squared_error")

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

    def greedy_action(self, x, verbose=False):
        Q = self.model.predict(np.array([x]))
        if verbose:
            print("Q = {}".format(Q))
            print("greedy = {}".format(np.argmax(Q)))
        return np.argmax(Q)


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
            if not self.rewards[pos - 1] < 0:
                pos = pos - 1
        return np.stack(to_be_stacked, axis=-1), rewards

    def minibatch(self, size, show=False):
        indices = random.sample(range(4, len(self.actions) - 1), size)
        x = np.zeros((size,) + self.screen_shape + (4,))
        y = np.zeros((size,) + self.screen_shape + (4,))
        d = np.zeros((size,))
        A = np.zeros((size,))
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
        fig.show()
        input("Press Enter to continue...")


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""MAIN"""""""""""""""""""

if __name__ == "__main__":
    dqn = DQN()
    dqn.create_model(plot=True, name='mybasicnetwork')
    dqn.save_model('my_basic_network_')
