from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten
from skimage.color import rgb2gray
from skimage.transform import resize
import pickle
import numpy as np
import random
import matplotlib.pyplot as plt
import pygame
from ple.games.flappybird import FlappyBird
from ple import PLE
from time import time

UP = 119
ACTIONS = dict()
ACTIONS[0] = 0
ACTIONS[1] = UP


class DQN:
    def __init__(self):
        self.dqn = None
        self.mini_batch_size = 32
        self.a = dict()
        self.r = dict()
        self.s = dict()
        self.game_id = None
        self.shape = 0
        self.state = []
        self.max_init_game = 0

    def create_dqn(self):
        self.dqn = Sequential()
        # 1st layer
        self.dqn.add(Conv2D(filters=16, kernel_size=(8, 8), strides=4, activation="relu", input_shape=(84, 84, 4)))
        # 2nd layer
        self.dqn.add(Conv2D(filters=32, kernel_size=(4, 4), strides=2, activation="relu"))
        self.dqn.add(Flatten())
        # 3rd layer
        self.dqn.add(Dense(units=256, activation="relu"))
        # output layer
        self.dqn.add(Dense(units=2, activation="linear"))

        self.dqn.compile(optimizer="rmsprop", loss="mean_squared_error")

    def make_training_set(self, folder):
        filehander = open(folder + "/images.pickle", "rb")
        images = pickle.load(filehander)
        filehander.close()

        filehander = open(folder + "/rewards.pickle", "rb")
        rewards = pickle.load(filehander)
        filehander.close()

        filehander = open(folder + "/actions.pickle", "rb")
        actions = pickle.load(filehander)
        filehander.close()

        filehander = open(folder + "/game.pickle", "rb")
        game_id = pickle.load(filehander)
        filehander.close()
        self.game_id = game_id
        self.shape = np.shape(images[0])

        for i in set(game_id):
            self.r[i] = []
            self.a[i] = []
            self.s[i] = []

        for i, idd in enumerate(game_id):
            r = rewards[i]
            if r == 0:
                self.r[idd].append(0.1)
            elif r == -5:
                self.r[idd].append(-1)
            else:
                self.r[idd].append(1)

            self.a[idd].append(actions[i])
            self.s[idd].append(images[i])

        self.max_init_game = max(game_id)

    def save_dqn(self, name):
        self.dqn.save(name)

    @staticmethod
    def load_dqn(name):
        return load_model(name)

    def append_game(self, images, rewards, actions):
        if len(self.a) > 500:
            kk = 0
            while kk <= self.max_init_game:
                kk = random.choice(list(self.a.keys()))
            self.a.pop(kk)
            self.r.pop(kk)
            self.s.pop(kk)
            print("pop {}".format(kk))

        idx = max(self.a.keys()) + 1
        self.r[idx] = []
        self.a[idx] = []
        self.s[idx] = []
        for i in range(len(actions)):
            r = rewards[i]
            if r == 0:
                self.r[idx].append(0.1)
            elif r == -5:
                self.r[idx].append(-1)
            else:
                self.r[idx].append(1)

            self.a[idx].append(actions[i])
            self.s[idx].append(images[i])

    def get_mini_batch(self, size=32):
        idx = np.random.choice(self.game_id, size=size)
        print(len(idx))
        r = np.zeros(shape=(size, 1))
        a = np.zeros(shape=(size, 1), dtype=int)
        x = np.zeros((size,) + self.shape + (4,))
        y = np.zeros((size,) + self.shape + (4,))
        d = np.zeros(shape=(size, 1), dtype=int)

        for i in range(size):
            jmin = 0
            jmax = len(self.a[idx[i]])
            j = random.choice(range(jmin, jmax))
            r[i] = self.r[idx[i]][j]
            if r[i] == -1:
                d[i] = 1
            a[i] = self.a[idx[i]][j]

            x_list = []
            y_list = []
            for k in range(j, j - 4, -1):
                x_list = [self.s[idx[i]][max([0, k])]] + x_list
                y_list = [self.s[idx[i]][max([0, min(jmax - 1, k)])]] + y_list
            x[i] = np.stack(x_list, axis=-1)
            y[i] = np.stack(y_list, axis=-1)

        return x, a, r, y, d

    def learn(self, gamma=0.95, ite=1000):
        game = FlappyBird()
        p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=True, display_screen=True)
        # Note: if you want to see you agent act in real time, set force_fps to False. But don't use this setting for learning, just for display purposes.

        p.init()
        reward = 0.0
        UP = 119
        nb_games = 10
        count = 0
        for i in range(ite):
            p.reset_game()
            actions = []
            images = []
            rewards = []
            X = []
            while (not p.game_over()):
                count += 1
                state = game.getGameState()
                screen = p.getScreenRGB()
                s = DQN.process_screen(screen)
                images.append(s)

                if len(X) < 4:
                    X.append(s)
                    action = 0
                else:
                    X.append(s)
                    X.pop(0)
                    # action selection
                    if np.random.rand() < DQN.epsilon(count):
                        action = np.random.choice([0, UP])
                        print("Action from exploration = {}".format(action))
                    else:
                        x = np.stack(X, axis=-1)
                        action = ACTIONS[self.greedy_action(x)]
                        print("Action from dqn = {}".format(action))

                actions.append(action)

                reward = p.act(action)
                rewards.append(reward)

                # learning
                x, a, r, y, d = cnn.get_mini_batch()
                QY = self.dqn.predict(y)
                QYmax = QY.max(1).reshape((self.mini_batch_size, 1))
                update = r + gamma * (1 - d) * QYmax
                QX = self.dqn.predict(x)
                for j in range(self.mini_batch_size):
                    # print("              " + str(a[j][0])))
                    QX[j, a[j][0]] = update[j][0]
                loss = self.dqn.train_on_batch(x=x, y=QX)
                if count % 100 == 0:
                    print("Itération {}".format(count))
                    print("Losse {}".format(loss))

            for j, a in enumerate(actions):
                if a == UP:
                    actions[j] = 1
            self.append_game(images, rewards, actions)

    @staticmethod
    def epsilon(step):
        limit = 2000*50
        if step < limit:
            if step % 100 == 0:
                print("epsilon {}".format(.5 - step / limit * 0.4))
            return .5 - step / limit * 0.4
        return .1

    @staticmethod
    def process_screen(x):
        return 256 * resize(rgb2gray(x), (102, 100))[18:, :84]

    def greedy_action(self, x):
        Q = self.dqn.predict(np.array([x]))
        return np.argmax(Q[0])


if __name__ == "__main__":
    cnn = DQN()
    cnn.create_dqn()
    cnn.dqn = DQN.load_dqn("premier")
    cnn.make_training_set("data2")

    cnn.learn(ite=2000)
    cnn.save_dqn("test")
