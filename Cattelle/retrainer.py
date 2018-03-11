import keras
import numpy as np
from ple import PLE
from ple.games import FlappyBird
from tqdm import trange

from ExperienceReplay import ExperienceReplay
from config import Config as config
from utils import StateHolder


class Retrainer:
    """
        Retraining class, used to further train an existing Keras model
    """

    def __init__(self, model_file):
        """
            Load the existing model file
        Args:
            model_file (str): Path to the model file (h5 file)
        """

        self.model = keras.models.load_model(model_file)
        self.er = ExperienceReplay(config.ER_SIZE, config.HISTORY_LENGTH, config.MINIBATCH_SIZE)
        self.stateholder = StateHolder()

        game = FlappyBird(graphics="fixed")
        p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=True, display_screen=True)
        # Note: if you want to see you agent act in real time, set force_fps to False. But don't use this setting for
        # learning, just for display purposes.
        p.init()

        self.game = game
        self.p = p

        self._gen_er_samples()

    def train_network(self):
        """"
            Train the DQN according to the settings defined in config.py
        """
        print(len(self.er.memory))
        print('Starting training')

        p = self.p
        screen = p.getScreenRGB()

        # Main training loop, runs for NB_TIMESTEPS iterations
        for i in trange(config.TIMESTEPS):
            action = self.eps_greedy(i)

            reward = p.act(action)
            # Shape reward to include rewardAlive
            # This is awarded when the agent survives for one timestep (without passing a pipe)
            if reward == 0.0:
                reward = config.REWARD_ALIVE
            # Clip negative rewards so that the reward space remains in [-1,1]
            if reward < 0:
                reward = -1.0

            new_screen = p.getScreenRGB()
            done = p.game_over()

            self.er.append_sample(screen, action, reward, new_screen, done)

            state, a, r, new_state, D = self.er.minibatch()

            state = self._unpack_state(state)
            new_state = self._unpack_state(new_state)

            Q = self.model.predict(state)  # shape (minibatch_size, 2)
            new_Q = self.model.predict(new_state)  # shape (minibatch_size, 2)

            # row-wise maximum, shape (minibatch_size, )
            max_new_Q = new_Q.max(1).reshape((config.MINIBATCH_SIZE,))

            update = r + (1 - D) * (config.DISCOUNT_RATE * max_new_Q)

            Q[:, (a // 119).astype(int)] = update.reshape(config.MINIBATCH_SIZE, 1)

            # Incremental training
            self.model.train_on_batch(x=state, y=Q)

            if i % config.TEST_DELTA == 0 and i > 0:
                print('Testing the network...')
                mean_score, max_score = self.eval_network(config.NUM_TEST_TRIALS)
                print('Current scores for the network:\n',
                      f'\tmean -> {mean_score}'
                      f'\tmax -> {max_score}')

            if i % config.SAVE_DELTA == 0 and i > config.MIN_ER_SIZE:
                print('Saving network...')
                self._write_network(config.MODEL_FILENAME)

            if done:
                p.reset_game()

            screen = p.getScreenRGB()

        print('Training done, saving final weights')
        self._write_network(config.MODEL_FILENAME)

    def eps_greedy(self):
        """
            Epsilon-greedy explorator (GLIE). Takes a random action with probability epsilon (linearly decreasing
            from 1.0 to 0.1 over all timesteps), otherwise the greedy action from the current Q-network
        Returns:
            action (int)): The next action to make
        """
        # The epsilon parameter decreases linearly over all timesteps
        epsilon = 0.1  # Fixed eps during retraining

        if np.random.rand() <= epsilon:
            # Take random action, either None (0) or flap (119)
            action = np.random.choice([0, 119], p=[1 - config.PROB_FLAP, config.PROB_FLAP])
        else:
            state = self.er.memory[-1][3]
            state = self._unpack_state(state).reshape((1, config.HISTORY_LENGTH, 84, 84))
            # reshape necessary since dqn.predict expect a list of samples (in this case only a single sample)
            action_array = self.model.predict(state)
            action = action_array.argmax()
            action *= 119  # the argmax is either 0 or 1, whereas the correct actions are either 0 or 119

        return action

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
                action_array = self.model.predict(holder.get_dqn_input())
                action = action_array.argmax() * 119

                scores[i] += p.act(action)
                holder.append(p.getScreenRGB())

        return scores.mean(), scores.max()

    def _write_network(self, filename='weights.dqn'):
        """
            Save the full model (architecture + weights + status of the optimiser) to the HDF5 archive located at
            "filename"
        Args:
            filename (str): Location of saved model (path)
        """
        self.model.save(filename)

    def _unpack_state(self, state):
        """
            Unroll the state array (array of deques) along its 1st axis (i.e. the dequeue axis)

        Args:
            state (np.ndarray): State of deques to unpack, shape (n,)

        Returns:
            unpacked (np.ndarray): Unpacked state, ready to be fed to the DQN, shape (n, history_length, 84, 84)
        """
        return np.array([np.array(elt) for elt in state])

    def _gen_er_samples(self):
        """
            Use the existing model to generate enough sample to start training (i.e. MIN_ER_SIZE samples according to
            the config file)
        """
        print(f"Generating {config.MIN_ER_SIZE} samples using the existing model")

        self.stateholder.append(self.p.getScreenRGB())

        for i in trange(config.MIN_ER_SIZE):
            screen = self.p.getScreenRGB()

            action = self.model.predict(self.stateholder.get_dqn_input())
            action = action.argmax() * 119

            reward = self.p.act(action)

            # Shape reward exactly as we do during training
            if reward == 0.0:
                reward = config.REWARD_ALIVE
            if reward < 0.0:
                reward = -1.0

            new_screen = self.p.getScreenRGB()
            done = self.p.game_over()

            # Append to the stateholder
            self.stateholder.append(new_screen)

            # Append to the ER
            self.er.append_sample(screen, action, reward, new_screen, done)

            if done:
                self.p.reset_game()

        print(f'Successfully generated {config.MIN_ER_SIZE} samples')


if __name__ == '__main__':
    retrainer = Retrainer(config.MODEL_FILENAME)
    retrainer.train_network()
