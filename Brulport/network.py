from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten
from keras import optimizers, initializers
from skimage.color import rgb2gray
from skimage.transform import resize
import numpy as np
import random
from timeit import time
from ple.games.flappybird import FlappyBird
from ple import PLE

random.seed(time.time())

# Conversion between action of the game (0, 119) and action for the dqn (0, 1)
UP = 119
ACTIONS = dict()
ACTIONS[0] = 0
ACTIONS[1] = UP


class DQN:
    """
    Class which implements a DQN to play the Flappy Bird game.
    """

    def __init__(self):
        # Dqn parameters
        self.dqn = None  # the neural network
        self.mini_batch_size = 32
        self.limit_epsilon = 200000
        self.epsilon_init = 0.1
        self.epsilon_final = 0.001
        self.learning_rate = 0.00001
        self.iteration = 0
        self.SizeMemory = 100000
        self.probability_of_UP = 0.5
        self.iterations_before_train = 5000

        # Memory buffer
        self.index = -1  # index of the last inserted element
        self.CurrentSize = 0  # current size of the memory (~ # of iteration)
        self.shape = (84, 84)  # shape of the image passed as input
        # /!\ to save memory, we keep only the screen, not the entire state (4 frames)
        self.states = np.zeros(shape=(self.SizeMemory,) + self.shape, dtype=np.uint8)
        self.actions = np.zeros(shape=(self.SizeMemory,), dtype=np.uint8)
        self.rewards = np.zeros(shape=(self.SizeMemory,))

    def create_dqn(self):
        """
        Create the initial neural network
        """
        self.dqn = Sequential()
        # Initialize all the weights with a normal distribution of std 0.01
        initializer = initializers.RandomNormal(mean=0, stddev=0.01, seed=None)

        # 1st layer
        self.dqn.add(Conv2D(filters=16, kernel_size=(8, 8), strides=4, activation="relu", input_shape=(84, 84, 4),
                            kernel_initializer=initializer))

        # 2nd layer
        self.dqn.add(Conv2D(filters=32, kernel_size=(4, 4), strides=2, activation="relu",
                            kernel_initializer=initializer))
        self.dqn.add(Flatten())

        # 3rd layer
        self.dqn.add(Dense(units=256, activation="relu", kernel_initializer=initializer))
        # Output layer
        self.dqn.add(Dense(units=2, activation="linear", kernel_initializer=initializer))

        # Using the RMS prop optimizer with a learning rate of 1e-5
        optimizer = optimizers.RMSprop(lr=self.learning_rate)
        self.dqn.compile(optimizer=optimizer, loss="mean_squared_error")

    def save_dqn(self, name):
        """
        Useful to save a NN in the network folder.
        :param name: the NN name
        """
        self.dqn.save("networks/" + name)

    @staticmethod
    def load_dqn(name):
        """
        Load a NN from the network folder.
        :param name: the NN name
        :return: the NN
        """
        return load_model("networks/" + name)

    def append_iteration(self, state, action, reward):
        """
        Save one iteration in the memory buffer. The future state is not saved since it can be retrieve \
        from the next iteration.
        :param state: the screen from the iteration
        :param action: the action taken from this screen
        :param reward: the obtained reward after the action is applied
        """
        # Convert the value 119 to 1 for the dqn
        if action == UP:
            action = 1

        # Index of the future iteration, must not exceed the size of the memory
        self.index = (self.index + 1) % self.SizeMemory
        self.CurrentSize += 1
        self.states[self.index, :] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward_model(reward)

    def save_memory(self):
        """
        Save the memory buffer in the data folder. Useful to stop the learning and to continue later.
        """
        np.save("data/states", self.states)
        np.save("data/actions", self.actions)
        np.save("data/rewards", self.rewards)
        np.save("data/memory_state", np.array([self.CurrentSize, self.index]))

    def load_memory(self):
        """
        Load the last memory buffer from the data folder.
        :return:
        """
        self.states = np.load("data/states.npy")
        self.actions = np.load("data/actions.npy")
        self.rewards = np.load("data/rewards.npy")
        memory_state = np.load("data/memory_state.npy")
        self.CurrentSize = memory_state[0]
        self.index = memory_state[1]

    def get_mini_batch(self, size=32):
        """
        Choose random (x, a, r, y, d) samples in the memory buffer to update the NN.
        :param size: Size of the minibatch
        :return: Five arrays: x, a, r, y, d
        """
        x = np.zeros((size,) + self.shape + (4,), dtype=np.uint8)
        a = np.zeros(shape=(size, 1), dtype=np.uint8)
        r = np.zeros(shape=(size, 1))
        y = np.zeros((size,) + self.shape + (4,), dtype=np.uint8)
        d = np.zeros(shape=(size, 1), dtype=np.uint8)  # stock if the state is terminal

        for i in range(size):
            # Random index in the memory buffer
            j = random.randint(4, self.CurrentSize - 1) % self.SizeMemory - 1
            a[i] = self.actions[j]
            r[i] = self.rewards[j]
            if r[i] == -1:  # in this case, 'y' isn't used because the state is terminal
                d[i] = 1

            # The state is made of 4 screen
            x_list = []
            y_list = []
            idx = j
            for k in range(4):
                x_list = [self.states[idx]] + x_list
                y_list = [self.states[idx + 1]] + y_list
                idx_new = idx - 1
                # We append the previous screen if the latter is not a terminal state, \
                # in the other case we pad with the actual screen
                if self.rewards[idx_new] != -1:
                    idx = idx_new

            x[i] = np.stack(x_list, axis=-1)
            y[i] = np.stack(y_list, axis=-1)

        return x, a, r, y, d

    def eval_dqn(self, p_game, nb_games=20):
        """
        Used to evaluate the performance of the NN during the learning phase.
        :param p_game: the PLE game
        :param nb_games: Number of game used to estimate the average score of the NN.
        :return: the average score and the maximum score obtained during the test.
        """
        # Buffer to stock the state
        X = []
        scores = np.zeros(nb_games)
        for i in range(nb_games):
            p_game.reset_game()
            while not p_game.game_over():
                s = process_screen(p_game.getScreenRGB())
                # for the very first game, we wait to have a full state before taking action from the NN
                if len(X) < 4:
                    action = UP
                    X.append(s)
                else:
                    X.append(s)
                    X.pop(0)
                    x = np.stack(X, axis=-1)
                    action = ACTIONS[greedy_action(self.dqn, x)]

                # play the action and save the reward
                reward = p_game.act(action)

                # update the score
                scores[i] = scores[i] + reward

        average_score = np.mean(scores)
        max_score = np.max(scores)
        return average_score, max_score

    def learn(self, gamma=0.95, ite=1000):
        """
        Perform the learning phase of the NN.
        :param gamma: discount rate of the reward
        :param ite: number of iterations to reach
        :return: the mean (and max) loss very 100 iterations, \
        the average (and max) score evaluated every 25000 evaluation
        """
        # Game initialisation
        game = FlappyBird(graphics="fixed")
        p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=True, display_screen=True)
        p.init()

        # Variable to keep the results
        losses = []
        mean_losses = []
        max_losses = []
        average_scores = []
        max_scores = []
        evaluation = False

        # Buffer to stock the state
        X = []

        while self.iteration < ite:
            p.reset_game()

            while not p.game_over():
                self.iteration += 1
                if self.iteration % 100 == 0:
                    print("\n--------------------Iteration {}------------------------".format(self.iteration))

                # Retrieve and process the screen-------------------
                screen = p.getScreenRGB()
                state = process_screen(screen)

                # Action selection----------------------------------

                if len(X) < 4:
                    # For the very first game...
                    X.append(state)
                    action = 0
                else:
                    X.append(state)
                    X.pop(0)

                    # Compute the epsilon with respect to the number of iterations
                    eps = self.epsilon()
                    if self.iteration % 100 == 0:
                        print("Epsilon {}".format(eps))

                    # Random action
                    if np.random.rand() < eps:
                        if np.random.rand() < self.probability_of_UP:
                            action = UP
                        else:
                            action = 0

                    # Greedy action from the NN
                    else:
                        x = np.stack(X, axis=-1)
                        action = ACTIONS[greedy_action(self.dqn, x)]

                # Play the action, get the reward and append the sample (x, a, r) in the memory buffer
                reward = p.act(action)
                self.append_iteration(state, action, reward)

                # Learning phase -----------------------------------------------------
                if self.iteration > self.iterations_before_train:

                    # Get some random sample from the memory buffer
                    x, a, r, y, d = cnn.get_mini_batch()

                    # Prediction for each states and next states of the batch
                    QX = self.dqn.predict(x)
                    QY = self.dqn.predict(y)
                    # Maximum Q-value obtained for each next states
                    QYmax = QY.max(1).reshape((self.mini_batch_size, 1))
                    # Temporal difference
                    update = r + gamma * (1 - d) * QYmax

                    # Update the state prediction with the temporal difference
                    for j in range(self.mini_batch_size):
                        QX[j, a[j][0]] = update[j][0]

                    # Train the NN and save the loss value
                    losses.append(self.dqn.train_on_batch(x=x, y=QX))

                    # Print and save losses every 100 iterations
                    if self.iteration % 100 == 0:
                        l_mean = np.mean(losses)
                        l_max = np.max(losses)
                        losses = []
                        mean_losses.append(l_mean)
                        max_losses.append(l_max)
                        print("Mean loss on last 100 iterations: {:.5}".format(l_mean))
                        print("Max loss on last 100 iterations: {:.5}".format(l_max))

                # Performance evaluation every 25 000 iterations
                if self.iteration % 25000 == 0:
                    evaluation = True

            # Do the evaluation if it is the moment
            if evaluation:
                print("\n\n############### Evaluation DQN, ite: {} ################".format(self.iteration))
                average_score, max_score = self.eval_dqn(p)

                # Print and save the score
                average_scores.append(average_score)
                max_scores.append(max_score)
                print("Mean score: {}".format(average_score))
                print("Max score: {}\n\n".format(max_score))
                evaluation = False

        return mean_losses, max_losses, average_scores, max_scores

    def epsilon(self):
        """
        Compute the epsilon with the respect to the the DQN parameters and the actual number of iteration
        :return: epsilon
        """
        return max([self.epsilon_init - self.iteration / self.limit_epsilon * (self.epsilon_init - self.epsilon_final),
                    self.epsilon_final])


def reward_model(r):
    """
    Reward model used to clip the rewards. Note that it increase the reward for staying alive.
    :param r: the raw reward
    :return: the clipped reward
    """
    if r == 0:
        return 0.1
    elif r == -5:
        return -1.
    else:
        return 1.


def greedy_action(model, x):
    """
    Computes which action is the best for a specific state.
    :param model: the NN
    :param x: the state we want to evaluate
    :return: the best action to take (0 or 1)
    """
    Q = model.predict(np.array([x]))
    return np.argmax(Q[0])


def process_screen(x):
    """
    Converts the raw screen in gray scale and crops the useless border
    :param x: the raw screen
    :return: the processed screen
    """
    return np.array(256 * resize(rgb2gray(x), (102, 100))[18:, :84], dtype=np.uint8)


if __name__ == "__main__":
    cnn = DQN()
    new_network = False

    # If we create a network from scratch
    if new_network:
        cnn.create_dqn()
        cnn.iteration = 0
        old_mean_losses = []
        old_max_losses = []
        old_average_scores = []
        old_max_scores = []

    # If we continue a learning phase
    else:
        cnn.dqn = DQN.load_dqn("200 000_ite")
        cnn.iteration = 100000
        data = np.load("data/losses.npy")
        old_mean_losses = data[0]
        old_max_losses = data[1]
        old_average_scores = data[2]
        old_max_scores = data[3]
        cnn.load_memory()

    mean_losses, max_losses, average_scores, max_scores = cnn.learn(gamma=0.95, ite=200000)

    # Append results and save
    mean_losses = old_mean_losses + mean_losses
    max_losses = old_max_losses + max_losses
    average_scores = old_average_scores + average_scores
    max_scores = old_max_scores + max_scores
    np.save("data/losses", np.array([mean_losses, max_losses, average_scores, max_scores]))

    # Save memory and DQN
    cnn.save_memory()
    cnn.save_dqn("200 000_ite")
