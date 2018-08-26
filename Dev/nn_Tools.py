from collections import deque
from keras.layers import Dense, Conv2D, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np
import COUSIN.nn_PARAMS as params


def clip_reward(r):
    """ Reward according to each state obtained
    - if alive : 0.5
    - if dead : 0
    - if a pipe is passed : 1

    WARNING: be aware that this reward is stored in unit8. In other words, no negative numbers are advised.

    :param  r
    :return modified reward
    """
    if r > 0:   # pipe passed
        return 1
    return 0.1  # alive


def epsilon(step):
    """ Value of epsilon for each step. It presents three phases:
    - Constant value for INITIALIZATION steps (totally random)
    - Decreasing values from EPISLON0 to FINAL_EPSILON in EXPLORATION_STEPS
    - Constant value (FINAL_EPSILON)

    :param step
    :return epsilon
    """
    if step < params.INITIALIZATION:
        return 1
    elif step < params.EXPLORATION_STEPS:
        gradient = (params.FINAL_EPSILON - params.EPSILON0) / (params.EXPLORATION_STEPS - params.INITIALIZATION)
        return params.EPSILON0 - params.INITIALIZATION*gradient + step*gradient
    else:
        return params.FINAL_EPSILON


def greedy_action(model, x):
    """ Selection of the action needed by a greedy algorithm.

    :param model (network)
    :param x (screen modified)
    :return action (no jump (0) or jump (1))
    """
    q = model.predict(np.array([x]))
    return np.argmax(q)


def create_nn():
    """ Creation of the Neural Network
    Use of the optimizer Adam. Might have some problems using it if not the newest version of Keras/Tensorflow.
    Problem seen: http://forums.halite.io/t/keras-tensorflow-adam-optimizer-issue/1350/3

    :return model (Network, nn)
    """
    model = Sequential()
    # 1st layer
    model.add(Dense(units=8, activation="relu", input_dim=params.SIZE_STATE))
    # 2nd layer
    model.add(Dense(units=4, activation="relu"))
    # output layer
    model.add(Dense(units=2, activation="linear"))
    model.compile(optimizer=Adam(lr=params.LEARNING_RATE), loss="mean_squared_error")
    return model


class MemoryBuffer:
    """An experience replay buffer using numpy arrays"""

    def __init__(self, length, state_shape, action_shape):
        self.length = length
        self.state_shape = state_shape
        self.action_shape = action_shape
        shape = (length,) + (state_shape,)
        self.state_x = np.zeros(shape, dtype=np.int16)  # starting states
        self.state_y = np.zeros(shape, dtype=np.int16)  # resulting states
        shape = (length,) + action_shape
        self.actions = np.zeros(shape, dtype=np.uint8)  # actions
        self.rewards = np.zeros((length, 1), dtype=np.uint8)  # rewards
        self.terminals = np.zeros((length, 1), dtype=np.bool)  # true if resulting state is terminal
        self.terminals[-1] = True
        self.index = 0  # points one position past the last inserted element
        self.size = 0  # current size of the buffer

    def append(self, statex, a, r, statey, d):
        self.state_x[self.index] = statex
        self.actions[self.index] = a
        self.rewards[self.index] = r
        self.state_y[self.index] = statey
        self.terminals[self.index] = d
        self.index = (self.index + 1) % self.length
        self.size = np.min([self.size + 1, self.length])

    def stacked_state_x(self, index):
        im_deque = deque(maxlen=4)
        pos = index % self.length
        for i in range(4):
            im = self.state_x[pos]
            im_deque.appendleft(im)
            test_pos = (pos - 1) % self.length
            if not self.terminals[test_pos]:
                pos = test_pos
        return np.stack(im_deque, axis=-1)

    def stacked_state_y(self, index):
        im_deque = deque(maxlen=4)
        pos = index % self.length
        for i in range(4):
            im = self.state_y[pos]
            im_deque.appendleft(im)
            test_pos = (pos - 1) % self.length
            if not self.terminals[test_pos]:
                pos = test_pos
        return np.stack(im_deque, axis=-1)

    def minibatch(self, size):
        # return np.random.choice(self.data[:self.size], size=sz, replace=False)
        indices = np.random.choice(self.size, size=size, replace=False)
        x = np.zeros((size,) + (self.state_shape,) + (4,))
        y = np.zeros((size,) + (self.state_shape,) + (4,))
        for i in range(size):
            x[i] = self.stacked_state_x(indices[i])
            y[i] = self.stacked_state_y(indices[i])
        return x, self.actions[indices], self.rewards[indices], y, self.terminals[indices]

