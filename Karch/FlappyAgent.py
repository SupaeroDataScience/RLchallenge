import pickle
import random
from collections import deque

import keras
import numpy as np
from keras.initializers import RandomNormal
from keras.layers import Dense, Conv2D, Flatten
from keras.models import Sequential
from keras.optimizers import rmsprop
# from ple import PLE
# from ple.games.flappybird import FlappyBird
from skimage.color import rgb2gray
from skimage.transform import resize

UP = 119


def load_object(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)


def process_screen(x):
    '''
    Transform the rgb image into grayscale image and resize it to 80x80.
    The transformation takes out the ground and the part of the screen which is behind flappybird
    :param np.array x: screen to process.
    :return: processed screen.
    :rtype: np.array
    '''
    return np.array(256 * resize(rgb2gray(x), (102, 100))[18:102, :84], dtype=np.uint8)
    # return 256 * resize(rgb2gray(x), (102, 100))[18:102, :84]


class FlappyAgent:
    """
    DQN agent

    :ivar np.shape() state_size: shape of the input.
    :ivar int action_size: number of actions.
    :ivar deque() memory: memory as a list.
    :ivar float gamma: Discount rate.
    :ivar float epsilon: exploration rate.
    :ivar float epsilon_min: minimum exploration rate.
    :ivar float epsilon_decay: decay of the exploration rate.
    :ivar_float learning_rate: initial learning rate for the gradient descent
    :ivar keras.model model: neural network model
    """

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.state_memory = deque(maxlen=4)
        self.gamma = 0.95  # discount rate
        self.epsilon = 0.1  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = (0.1 - self.epsilon_min) / 300000
        self.learning_rate = 0.00001
        self.model = self._build_model()
        self.batch_size = 32

    def _build_model(self):
        """
        Build the different layers of the neural network.

        :return: The model of the neural network.
        """
        init = RandomNormal(mean=0.0, stddev=0.01, seed=None)
        model = Sequential()
        # 1st layer
        model.add(Conv2D(filters=16, kernel_size=(8, 8), strides=4, activation="relu", input_shape=(84, 84, 4),kernel_initializer=init))
        # 2nd layer
        model.add(Conv2D(filters=32, kernel_size=(4, 4), strides=2, activation="relu", kernel_initializer=init))
        model.add(Flatten())
        # 3rd layer
        model.add(Dense(units=256, activation="relu", kernel_initializer=init))
        # output layer
        model.add(Dense(units=2, activation="linear", kernel_initializer=init))

        model.compile(loss='mse',
                      optimizer=rmsprop(lr=self.learning_rate),  # using the Adam optimiser
                      metrics=['accuracy'])  # reporting the accuracy
        return model

    def importMemory(self, filename):
        for elem in load_object(filename):
            self.memory.append(elem)

    def remember(self, screen, action, reward):
        """
            Append screen, action and reward to memory
        """
        if reward == 0:
            reward = 0.1
        if reward == -5:
            reward = -1

        if action == None:
            action = 0
        screen = process_screen(screen)

        self.memory.append((screen, action, reward))

    def generate_minibatch(self):
        batch = []
        for kk in range(self.batch_size):
            i = random.randint(3, len(self.memory) - 2)
            pos = i
            s_list = deque(maxlen=4)
            s_prim_list = deque(maxlen=4)
            for j in range(4):
                s_list.appendleft(self.memory[pos][0])
                s_prim_list.appendleft(self.memory[pos + 1][0])
                if self.memory[pos - 1][2] != -1:
                    pos = pos - 1

            if self.memory[i][2] == -1:
                s_prim_list[-1] = self.memory[i][0]

            state = np.reshape(np.stack(s_list, axis=-1), [1, self.state_size, self.state_size, 4])
            next_state = np.reshape(np.stack(s_prim_list, axis=-1), [1, self.state_size, self.state_size, 4])

            action = self.memory[i][1]
            reward = self.memory[i][2]

            batch.append((state, action, reward, next_state))

        return batch

    def replay(self):
        """
        Core of the algorithm --> Q update according to the current weight of the network.
        :param int batch_size: Batch size for the batch gradient descent.
        :return:
        """
        minibatch = self.generate_minibatch()
        X, Y = [], []

        for state, action, reward, next_state in minibatch:  # We iterate over the minibatch
            target = reward + self.gamma * \
                              np.amax(self.model.predict(next_state)[0])  # delta=r+amax(Q(s',a')
            if reward == -1:
                target = reward

            target_f = self.model.predict(state)
            target_f[0][int(action / UP)] = target

            X.append(state)
            Y.append(target_f)

        X = np.array(X)
        X = np.reshape(X, [self.batch_size, self.state_size, self.state_size, 4])

        Y = np.array(Y)
        Y = np.reshape(Y, [self.batch_size, self.action_size])
        scores = self.model.fit(X, Y, epochs=1,
                                verbose=0,
                                batch_size=self.batch_size)  # fit for the previous action value --> update the weights in the network
        loss = scores.history['loss']

        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon - self.epsilon_decay  # exploration decay
        return loss

    def FlappyPolicy(self, screen):
        x = process_screen(screen)
        self.state_memory.append(x)

        if len(self.state_memory) < 4:
            state = np.stack([x, x, x, x], axis=-1)
        else:
            state = np.stack(self.state_memory, axis=-1)

        state = np.reshape(state, [1, self.state_size, self.state_size, 4])
        act_values = self.model.predict(state)

        return UP * np.argmax(act_values[0])

    def act(self, screen):
        """
        Act ε-greedy with respect to the actual Q-value output by the network.
        :param state: State from which we want to use the network to compute the action to take.
        :return: a random action with probability ε or the greedy action with probability 1-ε.
        """
        if np.random.rand() <= self.epsilon:
            action = random.choice([0, UP])
        else:
            action = self.FlappyPolicy(screen)
        return action

    def evaluateScore(self, nb_games, p):
        """
        Method for the flappy agent to evaluate how good he is !

        :param nb_games: number of games for the evaluation
        :return:
        """

        p.init()
        reward = 0.0
        cumulated = np.zeros((nb_games))
        for i in range(nb_games):
            p.reset_game()

            while (not p.game_over()):
                screen = p.getScreenRGB()
                action = self.FlappyPolicy(screen)

                reward = p.act(action)
                cumulated[i] = cumulated[i] + reward

        return cumulated

    def saveModel(self, name):
        self.model.save(name)

    def loadModel(self, name):
        self.model = keras.models.load_model(name)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


'''
Defining a method to match with the imposed run.py
'''
agent = FlappyAgent(84, 2)
agent.load('results/network300000_alpha00001')


def FlappyPolicy(state, screen):
    return agent.FlappyPolicy(screen)
