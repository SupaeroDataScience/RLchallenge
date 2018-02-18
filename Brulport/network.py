from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten
from keras import optimizers, initializers
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
LIMIT = 100000
EPSILON_FINAL = 0.01
LEARNING_RATE = 0.00001


class DQN:
    def __init__(self):
        self.dqn = None
        self.mini_batch_size = 32
        self.a = []
        self.r = []
        self.s = []
        # self.game_id = None
        self.shape = 0
        self.state = []
        self.max_init_game = 0
        self.iteration = 0

    def create_dqn(self):
        self.dqn = Sequential()
        # 1st layer
        self.dqn.add(Conv2D(filters=16, kernel_size=(8, 8), strides=4, activation="relu", input_shape=(84, 84, 4)))
        # 2nd layer
        self.dqn.add(Conv2D(filters=32, kernel_size=(4, 4), strides=2, activation="relu"))
        self.dqn.add(Flatten())
        # 3rd layer
        self.dqn.add(Dense(units=256, activation="relu",
                           kernel_initializer=initializers.RandomNormal(mean=0, stddev=0.05, seed=None)))
        # output layer
        self.dqn.add(Dense(units=2, activation="linear",
                           kernel_initializer=initializers.RandomNormal(mean=0, stddev=0.05, seed=None)))

        optimizer = optimizers.RMSprop(lr=LEARNING_RATE)
        self.dqn.compile(optimizer=optimizer, loss="mean_squared_error")

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

        self.shape = np.shape(images[0])

        for i, idd in enumerate(game_id):
            self.r.append(reward_model(rewards[i]))
            self.a.append(actions[i])
            self.s.append(images[i])

        self.max_init_game = max(game_id)

    def save_dqn(self, name):
        self.dqn.save("networks/" + name)

    @staticmethod
    def load_dqn(name):
        return load_model("networks/" + name)

    def append_game(self, images, rewards, actions):
        if len(self.a) > 75000:
            for i in range(len(actions)):
                self.a.pop(0)
                self.r.pop(0)
                self.s.pop(0)

        for i in range(len(actions)):
            self.r.append(reward_model(rewards[i]))
            self.a.append(actions[i])
            self.s.append(images[i])

    def get_mini_batch(self, size=32):
        r = np.zeros(shape=(size, 1))
        a = np.zeros(shape=(size, 1), dtype=int)
        x = np.zeros((size,) + self.shape + (4,))
        y = np.zeros((size,) + self.shape + (4,))
        d = np.zeros(shape=(size, 1), dtype=int)

        for i in range(size):
            j = random.randint(4, len(self.a) - 2)
            r[i] = self.r[j]
            a[i] = self.a[j]

            if r[i] == -1:
                d[i] = 1
                y_list = [self.s[j]]
            else:
                y_list = [self.s[j + 1]]

            x_list = []
            pos = j
            for k in range(4):
                x_list = [self.s[pos]] + x_list
                pos_new = pos - 1
                if self.r[pos_new] != -1:  # not a terminal state
                    pos = pos_new

            pos = j
            for k in range(3):
                y_list = [self.s[pos]] + y_list
                pos_new = pos - 1
                if self.r[pos_new] != -1:  # not a terminal state
                    pos = pos_new

            x[i] = np.stack(x_list, axis=-1)
            y[i] = np.stack(y_list, axis=-1)

        return x, a, r, y, d

    def prelearn(self, gamma=0.95, ite=1000):
        # learning
        losses = []
        for i in range(ite):
            x, a, r, y, d = cnn.get_mini_batch()
            QY = self.dqn.predict(y)
            QYmax = QY.max(1).reshape((self.mini_batch_size, 1))
            update = r + gamma * (1 - d) * QYmax
            QX = self.dqn.predict(x)
            for j in range(self.mini_batch_size):
                # print("              " + str(a[j][0])))
                QX[j, a[j][0]] = update[j][0]
            loss = self.dqn.train_on_batch(x=x, y=QX)
            losses.append(loss)
            if i % 100 == 0:
                print("Itération {}".format(i))
                print("Losse {}".format(loss))

        return losses

    def eval_dqn(self, p_game, nb_games=20):
        X = []
        cumulated = np.zeros((nb_games))
        for i in range(nb_games):
            p_game.reset_game()
            while not p_game.game_over():
                s = process_screen(p_game.getScreenRGB())
                if len(X) < 4:
                    action = UP
                    X.append(s)
                else:
                    X.append(s)
                    X.pop(0)
                    x = np.stack(X, axis=-1)
                    y = greedy_action(self.dqn, x)
                    action = ACTIONS[y]

                reward = p_game.act(action)
                cumulated[i] = cumulated[i] + reward

        average_score = np.mean(cumulated)
        max_score = np.max(cumulated)
        return average_score, max_score

    def learn(self, gamma=0.95, ite=1000):
        game = FlappyBird(graphics="fixed")
        p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=True, display_screen=True)
        # Note: if you want to see you agent act in real time, set force_fps to False. But don't use this setting for learning, just for display purposes.

        # Initialisation-----------------------
        nb_games = 0
        p.init()
        count = self.iteration
        losses = []
        mean_losses = []
        max_losses = []
        average_scores = []
        max_scores = []
        X = []

        while count < ite:
            nb_games += 1
            p.reset_game()
            actions = []
            images = []
            rewards = []
            while not p.game_over():
                count += 1
                screen = p.getScreenRGB()
                s = process_screen(screen)
                images.append(s)

                if len(X) < 4:
                    X.append(s)
                    action = 0
                else:
                    X.append(s)
                    X.pop(0)
                    # action selection
                    if np.random.rand() < epsilon(count):
                        action = np.random.choice([0, 0, 0, 0, UP])
                        # print("Action from exploration = {}".format(action))
                    else:
                        x = np.stack(X, axis=-1)
                        action = ACTIONS[greedy_action(self.dqn, x)]
                        # print("Action from dqn = {}".format(action))

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
                losses.append(loss)
                if count % 100 == 0:
                    l_mean = np.mean(losses)
                    l_max = np.max(losses)
                    losses = []
                    mean_losses.append(l_mean)
                    max_losses.append(l_max)
                    print("Mean loss on last 100 iterations: {:.5}".format(l_mean))
                    print("Max loss on last 100 iterations: {:.5}".format(l_max))

            for j, a in enumerate(actions):
                if a == UP:
                    actions[j] = 1
            self.append_game(images, rewards, actions)

            if nb_games % 1000 == 0:
                print("\n############### Evaluation DQN, game: {} ################".format(nb_games))
                average_score, max_score = self.eval_dqn(p)
                average_scores.append(average_score)
                max_scores.append(max_score)
                print("Mean score: {}".format(average_score))
                print("Max score: {}".format(max_score))

        self.iteration = count
        return mean_losses, max_losses, average_scores, max_scores


def reward_model(r):
    if r == 0:
        return 0.1
    elif r == -5:
        return -1
    else:
        return 1


def greedy_action(model, x):
    Q = model.predict(np.array([x]))
    return np.argmax(Q[0])


def epsilon(step):
    if step < LIMIT:
        eps = 1. - step / LIMIT * (1 - EPSILON_FINAL)
        if step % 100 == 0:
            print("\n--------------------Iteration {}------------------------".format(step))
            print("Epsilon {}".format(eps))
        return eps
    return EPSILON_FINAL


def process_screen(x):
    return 256 * resize(rgb2gray(x), (102, 100))[18:, :84]


if __name__ == "__main__":
    cnn = DQN()
    new_network = True
    if new_network:
        cnn.create_dqn()
    else:
        cnn.dqn = DQN.load_dqn("5000_games")

    cnn.iteration = 0
    cnn.make_training_set("data")

    mean_losses, max_losses, average_scores, max_scores = cnn.learn(gamma=0.95, ite=1000)
    # plt.figure()
    # plt.plot(losses)
    # plt.yscale('log')

    filehander = open("data/results.pickle", "wb")
    pickle.dump([mean_losses, max_losses, average_scores, max_scores], filehander)
    filehander.close()

    cnn.save_dqn("test")
