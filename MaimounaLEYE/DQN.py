################################Libraries Importation#####################################
#Import Standard Library  
import numpy as np 

# Import Flappy Bird environment
from ple.games.flappybird import FlappyBird
from ple import PLE

# Imoort Keras library
from keras.models import load_model, Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam

# Import MemoryBuffer Class
from MemoryBuffer import MemoryBuffer

#Import our constant parameters and utilities
from constant import *
from utils import *


############################## Implementation of the DQN algorithm###################
class DeepQNetwork:
    
    PATH = 'model.h5'

    # Init the game
    def __init__(self, mode=Mode.PLAY):

        # Init the model
        self.model = None

        # If the chosen miode is TEST or PLAY, the we need to load our model 
        #otherwise we need to create it
        if mode == Mode.PLAY or mode == Mode.TEST:
            print("Model loaded")
        elif mode == Mode.TRAIN:
            self.model = self.create_model()
        else:
            print("Sorry, this mode doesn't exist !")


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

  # Play with a trained model. This is the function our Flappy Policy will be based on. 
  #↓ We will modify it in order to get the ACTION vector. 
    def play(self, n=1, file_path=None):

        # Init the game and the environment
        game, env = init_flappy_bird(mode=Mode.PLAY)

        # Load the model already trained
        model = load_model(file_path)
        img = np.zeros(IMG_SHAPE)

        # For playing mode, we are not only going to play one game, but n ones
        #in order to check if the model was correctly trained
        for _ in range(n):
            score = 0

            # Reset env
            env.reset_game()
            x = Stack(frame=img, length=NB_FRAMES)

            while not env.game_over():

                # Get current state
                S = transform_screen(frame=env.getScreenRGB())
                x.append(frame=S)

                # Predict an action to perform based on our trained model
                A = np.argmax(model.predict(np.expand_dims(x.stack, axis=0)))

                # Get reward
                R = env.act(ACTIONS[A])
                score += R

            # Print the score
            print('Score : {}'.format(score))

    # Load an existing model for PLAY or TEST mode
    def load_model(self, file_path=None):
        model = load_model(file_path)
        return model


    # Train our model
    def train(self, total_steps=100000, replay_memory_size=100000, mini_batch_size=32, gamma=0.99):

        # Init the game and the environment
        game, env = init_flappy_bird(mode=Mode.TRAIN)

        # Get the initial state/screen
        S = transform_screen(frame=env.getScreenRGB())

        # Initialize the stack
        x = Stack(frame=S, length=NB_FRAMES)

        # Initialize replay memory
        replay_memory = MemoryBuffer(replay_memory_size, IMG_SHAPE, ACTION_SHAPE)

        # Initial state for evaluation, these are the parameters of our DQN.
        #The less the evaluation period id, the longer, the learning will take. We chose 
        #25000
        evaluation_period = 25000
        epoch = 0
        nb_epochs = total_steps // evaluation_period

        # Store scores
        mean_scores = np.zeros((nb_epochs))
        max_scores = np.zeros((nb_epochs))

        # Deep Q-learning with experience replay
        for step in range(total_steps):

            # Evaluation
            if ((step + 1) % evaluation_period == 0 and step > evaluation_period):
                print('Evaluation started ')

                # Run 25 games
                mean_scores[epoch], max_scores[epoch] = self.MCeval(trials=25, gamma=gamma, env=env)

                # Save the training
                self.model.save(self.PATH)

                print('Score : {}/{} (mean/max)'.format(mean_scores[epoch], max_scores[epoch]))
                epoch += 1

            # Action selection
            # Exploration strategy
            A = self.random_action() if (np.random.random() < self.epsilon(step)) else self.greedy_action(x.stack)

            # Step and get reward
            R = self.clip_reward(env.act(ACTIONS[A]))

            if R == 1.0: #Means tht we have successfully passed a pipe
                print("Successful Pipe ")

            # Get new processed screen
            S_ = transform_screen(env.getScreenRGB())

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

            # Prepare next transition based on the fact that the game is over or not
            if env.game_over():
                env.reset_game()
                S = transform_screen(frame=env.getScreenRGB())
                x.reset(frame=S)
            else:
                S = S_
                x.append(frame=S)

            print('Step {} / {}'.format(step, total_steps))


        # Display results
        for j in range(len(mean_scores)-1):
            print('Period : {} - Avg : {} - Max : {}'.format(j, mean_scores[j], max_scores[j]))


    # Decrease policy for the probability of taking a random action
    def epsilon(self, step):
        if step < 5e3:
            return 1
        elif step < 1e6:
            return (0.1 - 5e3*(1e-3-0.1)/(1e6-5e3)) + step * (1e-3-0.1)/(1e6-5e3)
        else:
            return 1e-3


    # Definition of the new reward policy 
    def clip_reward(self, r):
        return r


    # Execute a random action
    def random_action(self): 
        return (np.random.random() < 0.1)


    # Execute a greedy action
    def greedy_action(self, x):
        Q = self.model.predict(np.array([x]))
        return np.argmax(Q)


    # Monte-Carlo evaluation 
    def MCeval(self, trials, gamma, env):

        # Init
        scores = np.zeros((trials))
        img = np.zeros(IMG_SHAPE)

        # Play a certain number of games, and compute mean and max scores
        for i in range(trials):

            # Reset env
            env.reset_game()
            x = Stack(frame=img, length=NB_FRAMES)

            while not env.game_over():

                # Get current state
                S = transform_screen(frame=env.getScreenRGB())
                x.append(frame=S)

                # Predict an action to perform
                A = np.argmax(self.model.predict(np.expand_dims(x.stack, axis=0)))

                # Get reward
                R = env.act(ACTIONS[A])
                scores[i] += R

        return np.mean(scores), np.max(scores)


# Main to train and test the algorithm. It is used only for testing and training purposes
if __name__ == '__main__':

    # Init the algorithm for training or test purposes depending on the chosen mode
    flappy = DeepQNetwork(mode=Mode.PLAY)

    # Run the training
    # flappy.train(total_steps=400000,  replay_memory_size=400000,mini_batch_size=32,gamma=0.99)

    # Play with the trained model
    flappy.play(n=1, file_path='./model.h5')
