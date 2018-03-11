# Keras stuff
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.models import load_model, Sequential
from keras.optimizers import RMSprop, sgd, Adam

# Standard
import math
import numpy as np
import random
import sys

# Flappy Bird environment
from ple.games.flappybird import FlappyBird
from ple import PLE


# Implementation of the Q learning algorithm
class DeepQLearning:

    # Constants
    ACTIONS = [None, 119]

    # The different states of the game
    #   1 - Flappy position and velocity
    #   2 - Distance to pipes
    #   3 - Next pipe info
    #   4 - Next next pipe info
    STATES = [
        'player_y', 'player_vel',
        'next_pipe_dist_to_player', 'next_next_pipe_dist_to_player',
        'next_pipe_top_y', 'next_pipe_bottom_y',
        'next_next_pipe_top_y', 'next_next_pipe_bottom_y',
    ]

    MODEL_PATH = 'model.h5'


    # Init the game
    def __init__(self, load_model=False):

        # Load or reate the model to use: a deep neural network
        self.model = self.load_model() if load_model else self.create_model()

        # Save our model a first time
        self.save_model()

        # Print useful data
        self.max = 0
        self.UP = 1
        self.DOWN = 0
        

    # Create the deep neural network model
    def create_model(self):

        model = Sequential()
        model.add(Dense(units=128, activation="relu", input_shape=(8,)))
        model.add(Dense(units=128, activation="relu"))
        model.add(Dense(units=2, activation="linear"))
        # We use Adam with a lower learning rate
        model.compile(optimizer=Adam(lr=1e-4), loss='mean_squared_error')

        return model

    # Load an existing model (Keras call)
    def load_model(self):
        try:
            model = load_model(self.MODEL_PATH)
            return model
        except IOError:
            print("{} not found!".format(self.MODEL_PATH))
            sys.exit()

    # Save our model
    def save_model(self):
        self.model.save(self.MODEL_PATH)

        # TODO : save the current state (replay, nb_games, epsilon)


    # Play with a model trained
    def play(self, n=1, file_path=None):

        # use "Fancy" for full background, random bird color and random pipe color,
        # use "Fixed" (default) for black background and constant bird and pipe colors.
        game = FlappyBird(graphics="fixed")

        # Note: if you want to see you agent act in real time, set force_fps to False.
        # But don't use this setting for learning, just for display purposes.
        env = PLE(game,
                  fps=30,
                  frame_skip=1,
                  num_steps=1,
                  force_fps=False,
                  display_screen=True)

        # Init the environment (settings, display...)
        env.init()

        # Load the model
        model = load_model(file_path)

        # Let's play n games, and see if the model is correctly trained
        for _ in range(n):
            env.reset_game()
            while not env.game_over():
                S = self.get_game_data(game)
                Q = model.predict(S, batch_size=1)
                A = np.argmax(Q[0])
                env.act(self.ACTIONS[A])


    # Train the neural network
    def train(self, nb_games=1000, gamma=0.9, epsilon=1, batchSize=40, bufferSize=80, experience_replay=True):

        # use "Fancy" for full background, random bird color and random pipe color,
        # use "Fixed" (default) for black background and constant bird and pipe colors.
        game = FlappyBird(graphics="fixed")

        # Note: if you want to see you agent act in real time, set force_fps to False.
        # But don't use this setting for learning, just for display purposes.
        env = PLE(game,
                  fps=30,
                  frame_skip=1,
                  num_steps=1,
                  force_fps=True,
                  display_screen=False)

        # Init the environment (settings, display...)
        env.init()

        # Init variables
        replay = []

        # Run a certain number of frames, and see if the training works
        for i in range(nb_games):

            cumulated = 0
            pipes = 0

            # First, reset the game and get the current states
            env.reset_game()

            # Get the current state (S) of the game
            S = self.get_game_data(game)

            # Run until the game goes on
            while not env.game_over():

                # 1) In s, choose a (GLIE actor)
                Q = self.model.predict(S, batch_size=1)

                # Exploration / exploitation strategy
                A = self.random_action() if (np.random.random() < epsilon) else self.greedy_action(Q)

                # 2) Take action, observe reward (R) and new state (S_)
                R = self.reward_policy(env.act(self.ACTIONS[A]))
                S_ = self.get_game_data(game)

                if R == 1.0:
                    pipes += 1

                cumulated += R

                # Are we using experience replay
                if not experience_replay:

                    Q_ = self.model.predict(S_, batch_size=1)
                    y = np.zeros((1, 2))
                    y[:] = Q[:]
                    y[0][A] = R if R < 0 else (R + gamma * np.max(Q_[0]))
                    self.model.fit(x=S, y=y, batch_size=1, epochs=1, verbose=False)

                else:

                    # Experience replay storage
                    replay.append((S, A, R, S_))

                    # No learning until the buffer is full
                    # When the buffer is full, proceed to the training
                    if len(replay) >= bufferSize:

                        # Keep the size of the buffer fixed (pop the oldest element)
                        replay.pop(0)

                        # Randomly sample our experience replay memory
                        if batchSize < bufferSize:
                            minibatch = random.sample(replay, batchSize)
                        else:
                            minibatch = replay

                        # Init the X and Y arrays
                        X_train = []
                        y_train = []

                        # Iterate through the batch
                        for (S, A, R, S_) in minibatch:

                            # Predict the q values for the old and new states
                            Q = self.model.predict(S, batch_size=1)
                            Q_ = self.model.predict(S_, batch_size=1)

                            # Init y
                            y = np.zeros((1, 2))
                            y[:] = Q[:]

                            # Check if the game is over or not, depending on the reward
                            # R < 0 : terminal state
                            y[0][A] = R if R < 0 else (R + gamma * np.max(Q_[0]))

                            # Update the training variables
                            X_train.append(S.reshape(8,))
                            y_train.append(y.reshape(2,))

                        # Convert the variables to np arrays
                        X_train = np.array(X_train)
                        y_train = np.array(y_train)

                        # Then, fit our model
                        self.model.fit(x=X_train, y=y_train, batch_size=batchSize, epochs=1, verbose=False)

                # 5) s <- s'
                S = S_

            # Update exploitation / exploration strategy
            if epsilon > 0.1:
                epsilon -= (1.5 / nb_games)
                print(epsilon)

            if pipes > self.max:
                self.max = pipes

            # Print the current status of the game
            print('Game : {} / {} | Score : {} | Max : {}'.format(i, nb_games, (cumulated+5), self.max))

            # Save the network
            if (i % 1000 == 0):
                print('Save model')
                self.save_model()


    # Choose a random action to execute
    def random_action(self):
        return self.UP if np.random.random() < 0.1 else self.DOWN


    # Choose a greedy action
    def greedy_action(self, Q):
        return np.argmax(Q[0])


    # Return a valid reward
    def reward_policy(self, r):
        return r


    # Get the current state of the game, as an np.array
    def get_game_data(self, game):
        state = game.getGameState()
        return np.array(list(state.values())).reshape(1, len(state))


# Main to train and test the algorithm
if __name__ == '__main__':

    # Init the algorithm for training or test purposes
    flappy = DeepQLearning(load_model=False)

    # Run the training
    #flappy.train(nb_games=10000, bufferSize=32, batchSize=16, experience_replay=True)
    flappy.train(nb_games=10000, experience_replay=False)

    # Play some games with trained models
    #flappy.play(n=50, file_path='./Models/test-max-45.h5')
