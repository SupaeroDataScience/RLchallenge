# Training file
# Initialise and train the DQN
import numpy as np
from keras import optimizers
from keras.layers import Dense, Conv2D, Flatten
from keras.models import Sequential
from ple import PLE
from ple.games.flappybird import FlappyBird
from tqdm import trange

from ExperienceReplay import ExperienceReplay
from config import DebugConfig as config
from utils import StateHolder


class Trainer:

    # IDEA: Add ability to reload previous model to improve learning
    def __init__(self):
        # Initialise DQN structure and weights, copy config settings to class attributes ?
        # IDEA: Follow more closely the architecture in the paper if needed
        dqn = Sequential()
        # 1st layer
        dqn.add(Conv2D(filters=16, kernel_size=(8, 8), strides=4, activation="relu",
                       input_shape=(config.HISTORY_LENGTH, 84, 84), data_format="channels_first"))
        # Input shape is (no_samples, history_length, 84, 84), thus channels first
        # 2nd layer
        dqn.add(Conv2D(filters=32, kernel_size=(4, 4), strides=2, activation="relu"))
        dqn.add(Flatten())
        # 3rd layer
        dqn.add(Dense(units=256, activation="relu"))
        # output layer
        dqn.add(Dense(units=2, activation="linear"))

        optimizer = optimizers.RMSprop(lr=config.LEARNING_RATE,
                                       decay=config.DECAY)
        dqn.compile(optimizer=optimizer, loss="mean_squared_error")

        self.dqn = dqn

        self.er = ExperienceReplay(config.ER_SIZE, config.HISTORY_LENGTH, config.MINIBATCH_SIZE)

        game = FlappyBird(graphics="fixed")
        p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=True, display_screen=True)
        # Note: if you want to see you agent act in real time, set force_fps to False. But don't use this setting for
        # learning, just for display purposes.
        p.init()

        self.game = game
        self.p = p

    def train_network(self):
        """"
            Train the DQN according to the settings defined in config.py
        """
        p = self.p
        screen = p.getScreenRGB()
        # IDEA: Implement target network to stabilise learning
        # IDEA: Implement frame skipping if needed (equivalent to training every n loops and paste the same actions n
        #  times ?)

        # Main training loop, runs for NB_TIMESTEPS iterations
        for i in trange(config.TIMESTEPS):
            action = self.eps_greedy(i)

            reward = p.act(action)
            # Shape reward to include rewardAlive
            # This is awarded when the agent survives for one timestep (without passing a pipe)
            if reward == 0.0:
                reward = config.REWARD_ALIVE

            new_screen = p.getScreenRGB()
            done = p.game_over()

            self.er.append_sample(screen, action, reward, new_screen, done)

            if i >= config.MIN_ER_SIZE:
                if i == config.MIN_ER_SIZE:
                    print('Minimum size for ER memory reached, learning process started')

                state, a, r, new_state, D = self.er.minibatch()

                state = self._unpack_state(state)
                new_state = self._unpack_state(new_state)

                Q = self.dqn.predict(state)  # shape (minibatch_size, 2)
                new_Q = self.dqn.predict(new_state)  # shape (minibatch_size, 2)

                # max_Q = Q.max(1).reshape((config.MINIBATCH_SIZE,))
                # row-wise maximum, shape (minibatch_size, 1)
                max_new_Q = new_Q.max(1).reshape((config.MINIBATCH_SIZE,))

                update = r + (1 - D) * (config.DISCOUNT_RATE * max_new_Q)

                Q[(a // 119).astype(int)] = update.reshape(config.MINIBATCH_SIZE, 1)

                # Incremental training
                self.dqn.fit(x=state, y=Q, verbose=False)

            if i % config.TEST_DELTA == 0 and i > 0:
                print('Testing the network...')
                mean_score, max_score = self.eval_network(config.NUM_TEST_TRIALS)
                print('Current scores for the network:\n',
                      f'\tmean -> {mean_score}'
                      f'\tmax -> {max_score}')

            if i % config.SAVE_DELTA == 0 and i > 0:
                print('Saving network...')
                self._write_network(config.MODEL_FILENAME)

            if done:
                p.reset_game()

            screen = p.getScreenRGB()

        print('Training done, saving final weights')
        self._write_network(config.MODEL_FILENAME)

    def eps_greedy(self, step):
        """
            Epsilon-greedy explorator (GLIE). Takes a random action with probability epsilon (linearly decreasing
            from 1.0 to 0.1 over all timesteps), otherwise the greedy action from the current Q-network
        Args:
            step (int): Current step number

        Returns:
            action (int)): The next action to make
        """
        # The epsilon parameter decreases linearly over all timesteps
        # TODO: Find better evolution if applicable
        epsilon = 1.0 - (0.90 / config.TIMESTEPS) * step

        if np.random.rand() <= epsilon:
            # Take random action, either None (0) or flap (119)
            action = np.random.choice([0, 119])
        else:
            state = self.er.memory[-1][3]
            state = self._unpack_state(state).reshape((1, config.HISTORY_LENGTH, 84, 84))
            # reshape necessary since dqn.predict expect a list of samples (in this case only a single sample)
            action_array = self.dqn.predict(state)
            action = action_array.argmax()
            action *= 119  # the argmax is either 0 or 1, whereas the correct actions are either 0 or 119

        return action

    # def test_er(self):
    #     er = self.er
    #
    #     for i in trange(10):
    #         screen = self.p.getScreenRGB()
    #         action = 0
    #         reward = self.p.act(action)
    #         new_screen = self.p.getScreenRGB()
    #         er.append_sample(screen, action, reward, new_screen, self.p.game_over())

    def eval_network(self, trials=20):
        """
            Evaluate current performances of network.
        Args:
            trials: Number of trials to perform. One trial is one full game, from initialisation to game over

        Returns:
            results (tuple): Tuple of (mean score, max score). The mean score is averaged over all trials
        """

        scores = np.zeros(trials)

        # Create a local copy of the simulator to prevent messing up the training simulator
        game = FlappyBird(graphics="fixed")
        p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=True, display_screen=True)
        # Note: if you want to see you agent act in real time, set force_fps to False. But don't use this setting for
        # learning, just for display purposes.
        p.init()

        for i in range(trials):
            p.reset_game()
            screen = p.getScreenRGB()
            holder = StateHolder()
            holder.append(screen)

            while not p.game_over():
                action_array = self.dqn.predict(holder.get_dqn_input())
                action = action_array.argmax() * 119

                scores[i] += p.act(action)

        return scores.mean(), scores.max()

    def _write_network(self, filename='weights.dqn'):
        """
            Save the full model (architecture + weights + status of the optimiser) to the HDF5 archive located at
            "filename"
        Args:
            filename (str): Location of saved model (path)
        """
        self.dqn.save(filename)

    def _unpack_state(self, state):
        """
            Unroll the state array (array of deques) along its 1st axis (i.e. the dequeue axis)

        Args:
            state (np.ndarray): State of deques to unpack, shape (n,)

        Returns:
            unpacked (np.ndarray): Unpacked state, ready to be fed to the DQN, shape (n, history_length, 84, 84)
        """
        return np.array([np.array(elt) for elt in state])


if __name__ == '__main__':
    trainer = Trainer()
    # trainer.test_er()
    trainer.train_network()
