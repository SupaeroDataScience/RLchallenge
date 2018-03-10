# Standard
import numpy as np

# Keras stuff
from keras.models import load_model, Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam

# MemoryBuffer
from MemoryBuffer import MemoryBuffer
from params import *
from utils import *

# Flappy Bird environment
from ple.games.flappybird import FlappyBird
from ple import PLE


# Implementation of the DQN algorithm
class DeepQNetwork:

    # TODO : remove ?
    MODEL_PATH = 'model.h5'

    # Init the game
    def __init__(self, mode=Mode.PLAY):

        # Init model
        self.model = None

        # TODO Play or test modes -> load the model. Train mode -> create it
        if mode == Mode.PLAY or mode == Mode.TEST:
            print('Hello')
            self.model = self.load_model('./model.h5')

            return None
        elif mode == Mode.TRAIN:
            self.model = self.create_model()
        else:
            print('Ce mode de jeu n\'existe pas. Veuillez le changer !')


    # Create our neural network
    def create_model(self):

        # Sequential model
        dqn = Sequential()
        # 1st layer
        dqn.add(Conv2D(filters=16, kernel_size=(8,8), strides=4, activation="relu", input_shape=INPUT_SHAPE))
        # 2nd layer
        dqn.add(Conv2D(filters=32, kernel_size=(4,4), strides=2, activation="relu"))
        dqn.add(Flatten())
        # 3rd layer
        dqn.add(Dense(units=256, activation="relu"))
        # output layer
        dqn.add(Dense(units=2, activation="linear"))

        # We use Adam with a lower learning rate
        dqn.compile(optimizer=Adam(lr=LEARNING_RATE), loss='mean_squared_error')

        return dqn


    # TODO Load an existing model
    def load_model(self, file_path):
        model = load_model(file_path)
        return model


    # Play with a trained model
    def play(self, n=1, file_path= MODEL_PATH):

        # Init the game and the environment
        game, env = init_flappy_bird(mode=Mode.PLAY)

        # Load the model
        model = load_model(file_path)

        img = np.zeros(IMG_SHAPE)

        # Let's play n games, and see if the model is correctly trained
        for _ in range(n):

            # Init score
            score = 0

            # Reset env
            env.reset_game()
            x = Stack(frame=img, length=STACK_SIZE)

            while not env.game_over():

                # Get current state
                S = process_screen(frame=env.getScreenRGB())
                x.append(frame=S)

                # Predict an action to perform
                A = np.argmax(model.predict(np.expand_dims(x.stack, axis=0)))

                # Get reward
                R = env.act(ACTIONS[A])
                score += R

            # Print the score
            print('Score : {}'.format(score))


    # Train our model
    def train(self, total_steps=100000, replay_memory_size=100000, mini_batch_size=32, gamma=0.99):

        # Init the game and the environment
        game, env = init_flappy_bird(mode=Mode.TRAIN, graphics= "Fixed")

        # Get the initial state/screen
        S = process_screen(frame=env.getScreenRGB())

        # Initialize the stack
        x = Stack(frame=S, length=STACK_SIZE)

        # Initialize replay memory
        replay_memory = MemoryBuffer(replay_memory_size, IMG_SHAPE, ACTION_SHAPE)

        # Initial state for evaluation
        evaluation_period = 10000
        epoch = 0
        nb_epochs = total_steps // evaluation_period

        # Store scores
        mean_scores = np.zeros((nb_epochs))
        max_scores = np.zeros((nb_epochs))

        # Deep Q-learning with experience replay
        for step in range(total_steps):

            # Evaluation
            if ((step + 1) % evaluation_period == 0 and step > evaluation_period):
                print('Starting evaluation...')

                # Run 20 games and get mean and max scores
                mean_scores[epoch], max_scores[epoch] = self.MCeval(trials=20, gamma=gamma, env=env)

                # Save the training
                self.model.save(self.MODEL_PATH)

                print('Score : {}/{} (mean/max)'.format(mean_scores[epoch], max_scores[epoch]))
                print('Evaluation done. Resume training...')
                epoch += 1

            # Action selection
            # Exploration / exploitation strategy
            A = self.random_action() if (np.random.random() < self.epsilon(step)) else self.greedy_action(x.stack)

            # Step and get reward
            R = self.clip_reward(env.act(ACTIONS[A]))

            if R == 1.0:
                print('****************************  Tuyau passé  ****************************')

            # Get new processed screen
            S_ = process_screen(env.getScreenRGB())

            # Fill the memory buffer
            replay_memory.append(S, A, R, S_, env.game_over())

            # Train
            if step > mini_batch_size:
                X,A,R,Y,D = replay_memory.minibatch(mini_batch_size)
                QY = self.model.predict(Y)
                QYmax = QY.max(1).reshape((mini_batch_size, 1))
                update = R + gamma * (1 - D) * QYmax
                QX = self.model.predict(X)
                QX[np.arange(mini_batch_size), A.ravel()] = update.ravel()
                self.model.train_on_batch(x=X, y=QX)

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


        # Display results
        for j in range(len(mean_scores)-1):
            print('Period : {} - Avg : {} - Max : {}'.format(j, mean_scores[j], max_scores[j]))


    # Decrease policy for epsilon (probability of taking a random action)
    def epsilon(self, step):
        # Explore randomly on the first 5k steps
        if step < 5e3:
            return 1
        # Then decrease linearly from 0.1 to 0.001 between
        # 5k and 1e6 steps
        elif step < 1e6:
            return (0.1 - 5e3*(1e-3-0.1)/(1e6-5e3)) + step * (1e-3-0.1)/(1e6-5e3)
        else:
            return 1e-3


    # Define a new reward policy to control the scale of global scores
    def clip_reward(self, r):
        return r


    # Execute a random action
    def random_action(self): # TODO : change policy?
        return (np.random.random() < 0.1)
        #return np.random.randint(0,2)


    # Execute a greedy action
    def greedy_action(self, x):
        Q = self.model.predict(np.array([x]))
        return np.argmax(Q)


    # Monte-Carlo evaluation TODO :refacto
    def MCeval(self, trials, gamma, env):

        # Init
        scores = np.zeros((trials))
        img = np.zeros(IMG_SHAPE)

        # Play a certain number of games, and compute mean and max scores
        for i in range(trials):

            # Reset env
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
    flappy = DeepQNetwork(mode=Mode.TRAIN)
  #  flappy.play(n=50,file_path='./model.h5')

    # Run the training
    flappy.train(total_steps=400000,
                 replay_memory_size=400000,
                 mini_batch_size=32,
                 gamma=0.99)

    # Play some games with trained models
