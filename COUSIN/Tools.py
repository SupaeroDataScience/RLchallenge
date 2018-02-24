from collections import deque
from keras.layers import Dense, Conv2D, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np
import PARAMS as params
from skimage.color import rgb2gray
from skimage.transform import resize


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
    if r < 0:   # dead
        return 0
    return 0.5  # alive


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


def process_screen(screen):
    """ Process the screen to be used by the Deep Q Network
    - Selection of the important area (cut)
    - Transform in a grey scaled image (grey)
    - Resize image (SIZE_IMG)

    :param screen (in RGB)
    :return screen to be used
    """
    screen_cut = screen[60:, 25:310, :]                         # cut
    screen_grey = 256 * (rgb2gray(screen_cut))                  # in gray
    output = resize(screen_grey, params.SIZE_IMG, mode='constant')     # resize
    return output


def greedy_action(model, x):
    """ Selection of the action needed by a greedy algorithm.

    :param model (network)
    :param x (screen modified)
    :return action (no jump (0) or jump (1))
    """
    q = model.predict(np.array([x]))
    return np.argmax(q)


def create_dqn():
    """ Creation of the Deep Q Network
    Use of the optimizer Adam. Might have some problems using it if not the newest version of Keras/Tensorflow.
    Problem seen: http://forums.halite.io/t/keras-tensorflow-adam-optimizer-issue/1350/3

    :return model (Network, dqn)
    """
    input_shape = (params.SIZE_IMG[0], params.SIZE_IMG[1], 4)
    model = Sequential()
    # 1st layer
    model.add(Conv2D(filters=16, kernel_size=(8, 8), strides=4, activation="relu", input_shape=input_shape))
    # 2nd layer
    model.add(Conv2D(filters=32, kernel_size=(4, 4), strides=2, activation="relu"))
    model.add(Flatten())
    # 3rd layer
    model.add(Dense(units=256, activation="relu"))
    # output layer
    model.add(Dense(units=2, activation="linear"))
    model.compile(optimizer=Adam(lr=params.LEARNING_RATE), loss="mean_squared_error")
    return model


def evaluation(p, network, epoch, trials=100, logfile="Save/logfile.txt"):
    scores = np.zeros(trials)
    shape_img = params.SIZE_IMG
    for i in range(trials):     # trials = nb of games played
        p.reset_game()
        frames = deque([np.zeros(shape_img), np.zeros(shape_img), np.zeros(shape_img), np.zeros(shape_img)], maxlen=4)

        while not p.game_over():
            screen = process_screen(p.getScreenRGB())
            frames.append(screen)
            x = np.stack(frames, axis=-1)

            a = greedy_action(network, x)   # 0 or 1
            scores[i] += p.act(params.LIST_ACTIONS[a])

    results_max = np.max(scores)
    results_mean = np.mean(scores)

    # append in the logfile
    with open(logfile, 'a') as f:
        f.write(str(epoch) + ',' + str(results_mean) + ',' + str(results_max) + '\n')

    return results_mean, results_max


class MemoryBuffer:
    """An experience replay buffer using numpy arrays"""

    def __init__(self, length, screen_shape, action_shape):
        self.length = length
        self.screen_shape = screen_shape
        self.action_shape = action_shape
        shape = (length,) + screen_shape
        self.screens_x = np.zeros(shape, dtype=np.uint8)  # starting states
        self.screens_y = np.zeros(shape, dtype=np.uint8)  # resulting states
        shape = (length,) + action_shape
        self.actions = np.zeros(shape, dtype=np.uint8)  # actions
        self.rewards = np.zeros((length, 1), dtype=np.uint8)  # rewards
        self.terminals = np.zeros((length, 1), dtype=np.bool)  # true if resulting state is terminal
        self.terminals[-1] = True
        self.index = 0  # points one position past the last inserted element
        self.size = 0  # current size of the buffer

    def append(self, screenx, a, r, screeny, d):
        self.screens_x[self.index] = screenx
        self.actions[self.index] = a
        self.rewards[self.index] = r
        self.screens_y[self.index] = screeny
        self.terminals[self.index] = d
        self.index = (self.index + 1) % self.length
        self.size = np.min([self.size + 1, self.length])

    def stacked_frames_x(self, index):
        im_deque = deque(maxlen=4)
        pos = index % self.length
        for i in range(4):
            im = self.screens_x[pos]
            im_deque.appendleft(im)
            test_pos = (pos - 1) % self.length
            if not self.terminals[test_pos]:
                pos = test_pos
        return np.stack(im_deque, axis=-1)

    def stacked_frames_y(self, index):
        im_deque = deque(maxlen=4)
        pos = index % self.length
        for i in range(4):
            im = self.screens_y[pos]
            im_deque.appendleft(im)
            test_pos = (pos - 1) % self.length
            if not self.terminals[test_pos]:
                pos = test_pos
        return np.stack(im_deque, axis=-1)

    def minibatch(self, size):
        # return np.random.choice(self.data[:self.size], size=sz, replace=False)
        indices = np.random.choice(self.size, size=size, replace=False)
        x = np.zeros((size,) + self.screen_shape + (4,))
        y = np.zeros((size,) + self.screen_shape + (4,))
        for i in range(size):
            x[i] = self.stacked_frames_x(indices[i])
            y[i] = self.stacked_frames_y(indices[i])
        return x, self.actions[indices], self.rewards[indices], y, self.terminals[indices]