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

total_steps = 300000  # Number of frames that will be generated
mini_batch_size = 32
gamma = 0.99
memory_size = 100000  # size of the memory replay
epsilon_init = 1  # initial epsilon for linear and exponential decay
epsilon_final = 0.01  # final epsilon for linear decay
prop_decay = 1  # proportion of total steps during which linear decay happens
epsilon_decay = (epsilon_init - epsilon_final) / (prop_decay * total_steps)  # decay per step for linear decay
learning_rate = 1e-5
epsilon_tau = 75000  # time constant of exponential decay
nb_games = 10  # number of game used to evaluate the agent performances
policy = "lin"  # policy used by the agent (epsilon greedy with linear of exponential decay)


def epsilon_exp(step):
    """
    Exponential decay of epsilon
    :param int step: current step
    :return: (float) corresponding epsilon
    """
    return epsilon_final + (epsilon_init - epsilon_final) * np.exp(-step / epsilon_tau)


def epsilon_action(step):
    """
    Exponential growth of UP action probability
    :param int step: current step
    :return: float
    """
    return 0.20 + 0.3 * (1 - np.exp(-0.4 * step / epsilon_tau))


def epsilon(step):
    """
    Linear decay of epsilon
    :param int step: current step
    :return: float
    """
    if step < total_steps * prop_decay:
        return epsilon_init - step * epsilon_decay
    return epsilon_final


def clip_reward(r):
    """
    Shapes the reward so that dying yields -1 instead of -5
    """
    if r < 0:
        r = -1
    return np.int8(r)


def process_screen(x):
    """
    Processes the screen : convert to grayscale, crop, resize and convert to uint8.
    """
    return (255 * resize(rgb2gray(x)[50:, :410], (84, 84))).astype("uint8")


class Agent:
    """
    Class implementing the DQN agent.
    """

    def __init__(self, policy, model=None, plot_eps=True):
        self.model = model
        self.memory = MemoryBuffer(memory_size)
        self.buffer_state = deque(maxlen=4)
        self.state = None

        # We set the greedy policy of the agent and plot the corresponding epsilons

        if policy == "exp_action":
            self.policy = self.e_greedy_exp_action
            if plot_eps:
                xx = np.arange(total_steps)
                plt.plot(xx, epsilon_exp(xx), color='r')
                plt.plot(xx, epsilon_action(xx), color='b')
                plt.show()
        elif policy == "lin":
            self.policy = self.e_greedy
            if plot_eps:
                xx = np.arange(total_steps)
                y = [epsilon(i) for i in xx]
                plt.plot(xx, y)

    def create_model(self, plot=False, name=''):
        # initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

        dqn = Sequential()
        # 1st layer
        dqn.add(Conv2D(filters=16, kernel_size=(8, 8), strides=4, activation="relu", input_shape=(84, 84, 4)))
        # 2nd layer
        dqn.add(Conv2D(filters=32, kernel_size=(4, 4), strides=2, activation="relu"))
        dqn.add(Flatten())
        # 3rd layer
        dqn.add(Dense(units=256, activation="relu"))
        # output layer
        dqn.add(Dense(units=2, activation="linear"))

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

    def e_greedy(self, step):
        if np.random.rand() < epsilon(step):
            if np.random.rand() < 0.5:
                a_idx = 0
            else:
                a_idx = 1
        else:
            a_idx = self.greedy_action(verbose=False)
        return a_idx

    def e_greedy_exp_action(self, step):
        if np.random.rand() < epsilon_exp(step):
            if np.random.rand() < epsilon_action(step):
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
        update = R + gamma * (1 - D) * QYmax
        QX = self.model.predict(X)
        QX[np.arange(mini_batch_size), A.ravel()] = update.ravel()
        loss = self.model.train_on_batch(x=X, y=QX)
        return loss

    def evaluate_perfs(self, p, ACTIONS):
        cumulated = np.zeros((nb_games))

        for i in range(nb_games):
            p.reset_game()
            while not p.game_over():
                screen = process_screen(p.getScreenRGB())
                self.update_state(screen)
                action = self.greedy_action()
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


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""MAIN"""""""""""""""""""

if __name__ == "__main__":
    dqn = Agent("lin")
    dqn.create_model(plot=True, name='my_basic_network')
    #dqn.save_model('my_basic_network_')
