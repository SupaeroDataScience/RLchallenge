# Standard
import numpy as np
import os.path
import sys

# Keras stuff
from keras.models import load_model, Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam

# MemoryBuffer, params and utils functions
from MemoryBuffer import MemoryBuffer
from params import *
from utils import *

# Flappy Bird environment
from ple.games.flappybird import FlappyBird
from ple import PLE


# Implementation of the DQN algorithm
class DeepQNetwork:

    # The name of the model to save for training purposes
    DEFAULT_MODEL_PATH = 'model.h5'

    def __init__(self, mode=Mode.PLAY, file_path=None):
        """
        Init the Deep Q-Network class (load or create the model)

        mode        -- the mode we want to use (default PLAY)
        file_path   -- the path of a trained model to load and play
        """

        # Init model
        self.model = None

        # For training purposes, create a new model and save it
        if mode == Mode.TRAIN:
            self.model = self.create_model()
            self.save_model()

        # For playing purposes, load an existing model
        if mode == Mode.PLAY:
            self.model = self.load_model(file_path=file_path)

        # If model is invalid, abort execution
        if self.model == None:
            print('Le modèle spécifié est invalide. Veuillez le changer !')
            sys.exit()


    def create_model(self):
        """
        Create our neural network model (Convolutions + fully-connected)
        """

        # Sequential model
        dqn = Sequential()
        # 1st convolution layer
        dqn.add(Conv2D(filters=16, kernel_size=(8,8), strides=4, activation="relu", input_shape=INPUT_SHAPE))
        # 2nd convolution layer
        dqn.add(Conv2D(filters=32, kernel_size=(4,4), strides=2, activation="relu"))
        # Flattent the results
        dqn.add(Flatten())
        # 3rd layer - dense
        dqn.add(Dense(units=256, activation="relu"))
        # output layer - dense - 2 units because we have two actions (flap or not)
        dqn.add(Dense(units=2, activation="linear"))
        # We use Adam with a lower learning rate
        dqn.compile(optimizer=Adam(lr=LEARNING_RATE), loss='mean_squared_error')

        return dqn


    def load_model(self, file_path=None):
        """
        Load an existing model

        file_path -- the path of the model to load
        """

        if file_path == None or not os.path.isfile(file_path):
            return None
        return load_model(file_path)


    def save_model(self, file_path=None):
        """
        Save the current model

        file_path -- the path of the model (default: 'model.h5')
        """
        if file_path == None:
            self.model.save(self.DEFAULT_MODEL_PATH)
        else:
            self.model.save(file_path)


    def play(self, nb_games):
        """
        Play games with the screen to see our Flappy bird in action

        nb_games -- the number of games to play
        """

        # Init the game and the environment
        game, env = init_flappy_bird(mode=Mode.PLAY)

        # Init an empty img
        img = np.zeros(IMG_SHAPE)

        # Init score
        scores = np.zeros((nb_games))

        # Let's play n games, and see if the model is correctly trained
        for i in range(nb_games):

            # Reset env
            env.reset_game()
            x = Stack(frame=img, length=STACK_SIZE)

            while not env.game_over():

                # Get current state
                S = process_screen(frame=env.getScreenRGB())
                x.append(frame=S)

                # Predict an action to perform
                A = np.argmax(self.model.predict(np.expand_dims(x.stack, axis=0)))

                # Take action and get reward
                R = env.act(ACTIONS[A])
                scores[i] += R

            # Print the score
            print('Game {} - Score : {}'.format(i, scores[i]))

        # Print the score
        print('Min : {} - Max : {} - Avg : {}'.format(np.min(scores), np.max(scores), np.mean(scores)))


    def train(self, total_steps=100000, replay_memory_size=100000, mini_batch_size=32, gamma=0.99):
        """
        Train the deep q-network, using the pixels of the image

        total_steps         -- the number of steps to run
        replay_memory_size  -- the size of the memory buffer
        mini_batch_size     -- the size of the minibatch to use
        gamma               -- the discount factor
        """

        # Init the game and the environment
        game, env = init_flappy_bird(mode=Mode.TRAIN)

        # Get the initial state/screen
        S = process_screen(frame=env.getScreenRGB())

        # Initialize the stack
        x = Stack(frame=S, length=STACK_SIZE)

        # Initialize the memory buffer
        replay_memory = MemoryBuffer(replay_memory_size, IMG_SHAPE, ACTION_SHAPE)

        # Use a second network to inscrease the stability of the learning phase
        target_model = self.load_model(self.DEFAULT_MODEL_PATH)

        # Transfer the weights of the DQN to the target every 2500 steps
        update_period = 2500

        # Initial state for evaluation
        evaluation_period = 20000
        epoch = 0
        nb_epochs = total_steps // evaluation_period

        # Store scores for each epoch
        mean_scores = np.zeros((nb_epochs))
        max_scores = np.zeros((nb_epochs))

        # Deep Q-network with experience replay
        for step in range(total_steps):

            # Evaluation
            if ((step + 1) % evaluation_period == 0 and step > evaluation_period):
                print('Starting evaluation...')

                # Run an evaluation on 15 games and get the mean and max scores
                mean_scores[epoch], max_scores[epoch] = self.evaluate(nb_games=15, env=env)

                # Save the model to the disk
                self.save_model()

                # Display some stats for the user
                print('Score : {}/{} (mean/max)'.format(mean_scores[epoch], max_scores[epoch]))
                print('Evaluation done. Resume training...')

                # Stop the training if a suitable score has been reached
                if mean_scores[epoch] > 200:
                    break

                # Then increment the counter
                epoch += 1

            # Action selection : Exploration (random) vs exploitation (greedy) strategy
            A = self.random_action() if (np.random.random() < self.epsilon(step)) else self.greedy_action(x.stack)

            # Take action and get reward
            R = self.clip_reward(env.act(ACTIONS[A]))

            # Notify the user if a pipe was passed
            if R == 1.0:
                print('**********************  Tuyau passé  ************************')

            # Get new processed screen
            S_ = process_screen(frame=env.getScreenRGB())

            # Append new info to memory buffer
            replay_memory.append(S, A, R, S_, env.game_over())

            # Then train the model if enough data has been collected
            if step > mini_batch_size:

                X,A,R,Y,D = replay_memory.minibatch(mini_batch_size)
                QY = target_model.predict(Y) # prediction with the target network
                QYmax = QY.max(1).reshape((mini_batch_size, 1))
                update = R + gamma * (1 - D) * QYmax
                QX = self.model.predict(X) # prediction with the current dqn
                QX[np.arange(mini_batch_size), A.ravel()] = update.ravel()
                self.model.train_on_batch(x=X, y=QX)

            # Update the target network on a periodic basis
            # "Generating the targets using an older set of parameters adds a
            #  delay between the time an update to Q is made and the time the
            #  update affects the targets yj, making divergence or oscillations
            #  much more unlikely." (cf DeepMind Nature article)
            if ((step + 1) % update_period == 0 and step > update_period):
                self.save_model()
                target_model = self.load_model(self.DEFAULT_MODEL_PATH)

            # Prepare next transition
            if env.game_over():
                # Game over -> reset episode
                env.reset_game()
                S = process_screen(frame=env.getScreenRGB())
                x.reset(frame=S)
            else:
                # keep going
                S = S_
                x.append(frame=S)

            print('Step {} / {}'.format(step, total_steps))


        # At last, display a summary of the different evaluations
        for j in range(len(mean_scores)-1):
            print('Period : {} - Avg : {} - Max : {}'.format(j, mean_scores[j], max_scores[j]))


    # Decrease policy for epsilon (probability of taking a random action)
    def epsilon(self, step):
        """
        Get the probabilty of taking a random action (exploration vs exploitation)
        Linear decrease from 1 to 0.1 until 600 000 steps, then 0.001

        step -- the current step in the training loop
        """
        if step < 6e5:
            return 1.0 - 1.5e-6 * step
        else:
            return 1e-3


    def clip_reward(self, r):
        """
        Define a reward policy for the training.
        (I chose to keep the initial one, the results were good enough)

        r -- the reward of the immediate action
        """
        return r


    def random_action(self):
        """
        Define the exploration policy to adopt.
        Taking an int between 0 and 1 was not very interesting. Indeed, the
        flappy tends to go up with such a policy. I chose a probabilty of going
        up of 0.1, which is quite balanced between up and down moves.

        """
        return (np.random.random() < 0.1)


    def greedy_action(self, x):
        """
        To exploit, we need to predict the next moves with the current model

        x -- the last 4 frames (in order to get the velocity information)
        """
        Q = self.model.predict(np.array([x]))
        return np.argmax(Q)


    def evaluate(self, nb_games, env):
        """
        Do an evaluation of the current training by playing a few number of
        games and getting meaningful scores (mean and max).

        nb_games    -- the number of games to play
        env         -- the game environment
        """

        # Init
        scores = np.zeros((nb_games))
        img = np.zeros(IMG_SHAPE)

        # Play a certain number of games, and compute mean and max scores
        for i in range(nb_games):

            # Reset env and init stack
            env.reset_game()
            x = Stack(frame=img, length=STACK_SIZE)

            while not env.game_over():

                # Get current state
                S = process_screen(frame=env.getScreenRGB())
                x.append(frame=S)

                # Predict an action to perform
                A = np.argmax(self.model.predict(np.expand_dims(x.stack, axis=0)))

                # Get reward
                R = env.act(ACTIONS[A])
                scores[i] += R

        return np.mean(scores), np.max(scores)


# Main to train and test the algorithm
if __name__ == '__main__':

    # Init the algorithm for training or test purposes

    # TRAIN
    #flappy = DeepQNetwork(mode=Mode.TRAIN)
    #flappy.train(total_steps=500000,
    #             replay_memory_size=500000,
    #             mini_batch_size=32,
    #             gamma=0.99)

    # Or PLAY
    flappy = DeepQNetwork(mode=Mode.PLAY, file_path="./Models/dqn.h5")
    flappy.play(nb_games=20)
