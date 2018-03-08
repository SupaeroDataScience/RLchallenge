################################################################################
# File: Train.py
# Author: Nikola LOHINSKI
################################################################################

import pickle
import numpy as np
import argparse
from random import random as rd

from os import path
from math import floor
from ple.games.flappybird import FlappyBird
from ple import PLE

DOWN = 0
UP = 1
# Possible actions. 119 is to go up, and whatever else is to fall.
ACTIONS = [0, 119]


class Athlete:
    """Feature engineered flappybird athlete."""
    def __init__(
            self,
            gamma=0.85,
            alpha=0.65,
            x_reduce=15,
            y_reduce=15,
            v_reduce=2
    ):
        """Constructor.
        Args:
            gamma <float>: gamma in [0,1] for Q learning update
            alpha <float>: alpha in [0,1] for gradient descent in Q learning
            x_reduce <int>: value of the X axis simplification
            y_reduce <int>: value of the Y axis simplification
            v_reduce <int>: value of the velocity simplification
        """
        self.gamma = gamma
        self.alpha = alpha
        self.x_reduce = x_reduce
        self.y_reduce = y_reduce
        self.v_reduce = v_reduce
        self.max = 0
        self.Q = dict()
        self.print_data = dict()

    def state2coord(self, state):
        """Transforms default state into a feature engineered state that is
        explorable finitely.
        Args:
            state <dict>: base state
        Returns:
            str: 'x,y,z' like string representing the simplified state
        """
        gap_top = state.get('next_pipe_top_y')
        bird_y = state.get('player_y')
        player_vel = state.get('player_vel')
        pipe_x = state.get('next_pipe_dist_to_player')
        return ','.join([
            str(floor(pipe_x / self.x_reduce)),
            str(floor((gap_top - bird_y) / self.y_reduce)),
            str(floor(player_vel / self.v_reduce))
        ])

    def train(self, episodes=1000):
        """Train the athlete.
        Args:
            episodes <int>: number of episodes to iterate over
        """
        # Initialize exploration only data
        epsilon = 1
        epsilon_decay = 1 / episodes
        jumprate = 0.1
        # For log purposes only
        self.print_data = dict({
            'hits': 0,
            'games_played': 1,
            'below_15': 0,
            'pipes_below_15': 0,
            'ep': 0,
            'pipes': 0,
            'episodes': episodes
        })
        # Start game
        game = FlappyBird()
        env = PLE(game,
                  fps=30,
                  frame_skip=1,
                  num_steps=1,
                  force_fps=True,
                  display_screen=False)
        env.init()
        for _ in range(episodes):
            self.print_data['ep'] += 1
            self.print_data['pipes'] = 0
            # Reset game
            env.reset_game()
            S = self.state2coord(game.getGameState())
            while not env.game_over():
                # Has the state been visited already ?
                if self.Q.get(S) is None:
                    self.Q[S] = [0, 0]
                # Exploration
                if rd() < epsilon:
                    # Using a jump rate to orient exploration
                    A = UP if rd() < jumprate else DOWN
                else:
                    # Reinforcement
                    A = np.argmax(self.Q.get(S))
                # Perform action and get reward
                r = env.act(ACTIONS[A])
                if r == 1.0:
                    # For log purposes only
                    self.print_data['pipes'] += 1
                # Biase reward to orient exploration
                R = self.biase_reward(r)
                S_ = self.state2coord(game.getGameState())
                # Perform Q update
                self.update_q(S, A, R, S_)
                # Change state
                S = S_
            # Decrease exploration rate
            epsilon -= epsilon_decay
            # For log purposes only
            self.print_status()

    def act(self, state):
        """Act upon a state. If state has not been visited, then apply falling
        policy.
        Args:
            state <dict>: base state
        Returns:
            int: index of the action to perform
        """
        S = self.state2coord(state)
        if self.Q.get(S) is None:
            return DOWN
        return np.argmax(self.Q[S])

    def biase_reward(self, r):
        """Biase the reward to orient the exploration.
        Args:
            r <float>: original reward
        Returns:
            float: modified reward
        """
        return (1 if r == 0. else 10) if r != -5. else -100

    def update_q(self, S, A, R, S_):
        """Update value function Q if new state has been visited.
        Args:
            S <str>: current state
            A <int>: index of the chosen action
            R <float>: reward of the current action
            S_ <str>: next state
        """
        if self.Q.get(S_) is not None:
            delta = R + self.gamma * np.max(self.Q[S_]) - self.Q[S][A]
            self.Q[S][A] += self.alpha * delta

    def print_status(self):
        """Print status of the game (purely for training logging purposes).
        """
        nb_pipes = self.print_data['pipes']
        epoch = self.print_data['ep']
        horizon = self.print_data['episodes']
        sum_nb_pipes = self.print_data['hits']
        nb_games = self.print_data['games_played']
        nb_games_under_15 = self.print_data['below_15']
        sum_nb_pipes_under_15 = self.print_data['pipes_below_15']
        if nb_pipes > self.max:
            print('\nNew max: {} pipes, on game {}'.format(nb_pipes, epoch + 1))
            print('In previous sequence:')
            print(' - Average: {} pipes'.format(
                np.round(sum_nb_pipes / nb_games, decimals=1)
            ))
            print(' - games < 15: {}/{} ({}%)'.format(
                nb_games_under_15,
                nb_games,
                np.round(100 * nb_games_under_15 / nb_games, decimals=1))
            )
            print(' - Average of games < 15: {} pipes'.format(
                np.round(sum_nb_pipes_under_15 / nb_games, decimals=1)
            ))
            print(' - Number of Q states: {}'.format(len(self.Q)))
            self.max = nb_pipes
            self.print_data['hits'] = nb_pipes
            self.print_data['pipes_below_15'] = 0
            self.print_data['games_played'] = 1
            self.print_data['below_15'] = 1 if nb_pipes < 15 else 0
        else:
            if nb_pipes < 15:
                self.print_data['pipes_below_15'] += nb_pipes
                self.print_data['below_15'] += 1
            self.print_data['hits'] += nb_pipes
            self.print_data['games_played'] += 1
        if self.print_data['ep'] % floor(
                horizon / 10 if horizon < 10000 else horizon / 100
        ) == 0:
            print('Games: {}/{} ({}%)'.format(
                epoch,
                horizon,
                (100 * epoch / horizon))
            )

    def save_model(self, file_path):
        """Save model to file at a given path.
        Args:
            file_path <str>: path of the file to save
        """
        if path.isfile(file_path):
            r = input('File {} already exists. Overwrite ? (y,[n]) '.format(file_path))
            if r != 'y':
                return
        to_save = dict({
            'Q': self.Q,
            'gamma': self.gamma,
            'alpha': self.alpha,
            'x_reduce': self.x_reduce,
            'y_reduce': self.y_reduce,
            'v_reduce': self.v_reduce
        })
        with open(file_path, 'wb+') as f:
            pickle.dump(to_save, f, pickle.HIGHEST_PROTOCOL)

    def load_model(self, file_path):
        """Load model from given file path.
        Args:
            file_path <str>: path to model
        """
        with open(file_path, 'rb') as f:
            saved = dict(pickle.load(f))
            self.Q = saved.get('Q')
            self.gamma = saved.get('gamma')
            self.alpha = saved.get('alpha')
            self.x_reduce = saved.get('x_reduce')
            self.y_reduce = saved.get('y_reduce')
            self.v_reduce = saved.get('v_reduce')

    def play(self, fast=True):
        """Use athlete to play.
        Args:
            fast <bool>: set to True if the screen should be hidden and speed
            enhanced
        """
        game = FlappyBird()
        env = PLE(game,
                  fps=30,
                  frame_skip=1,
                  num_steps=1,
                  force_fps=fast,
                  display_screen=not fast)
        env.init()
        pipes = []
        i = 0
        while i < 100:
            env.reset_game()
            pipes.append(0)
            while not env.game_over():
                A = self.act(game.getGameState())
                r = env.act(ACTIONS[A])
                if r == 1.:
                    pipes[-1] += 1
            if not fast:
                print('\n- Score: {} pipes'.format(pipes[-1]))
                print('- Played {} games'.format(len(pipes)))
                print('- Average score: {} pipes'.format(np.round(np.mean(pipes), decimals=1)))
            else:
                i += 1

        print('\n- Max score: {} pipes'.format(np.max(pipes)))
        print('- Games < 15 pipes: {}'.format(
            len(tuple(filter(lambda x: x < 15, pipes)))
        ))
        print('- Played {} games'.format(100))
        print('- Average score: {} pipes'.format(
            np.round(np.mean(pipes), decimals=1))
        )


# Command line HMI to control athlete
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-l', '--load',
        help='Load a specific state engineered model from a file',
        type=str
    )
    parser.add_argument(
        '-s', '--save',
        help='Save model at the end of training to a file',
        type=str
    )
    parser.add_argument(
        '-p', '--play',
        action='store_true',
        help='Play only. Requires a model to be loaded'
    )
    parser.add_argument(
        '-t', '--test',
        action='store_true',
        help='Test only over 100 episodes without display. Requires a model to be loaded'
    )
    parser.add_argument(
        '-e', '--epochs',
        help='Train over a given number of episodes. Default is 1000',
        type=int
    )
    parser.add_argument(
        '-g', '--gamma',
        help='Use a specific gamma rate for Q-learning. Default is 0.85',
        type=float
    )
    parser.add_argument(
        '-a', '--alpha',
        help='Use a specific alpha rate for Q updates. Default is 0.7',
        type=float
    )
    parser.add_argument(
        '-x', '--x-reduce',
        help='Simplification of the state on X axis. Default is 15',
        type=int
    )
    parser.add_argument(
        '-y', '--y-reduce',
        help='Simplification of the state on Y axis. Default is 10',
        type=int
    )
    parser.add_argument(
        '-v', '--v-reduce',
        help='Simplification of the state on velocity axis. Default is 2',
        type=int
    )
    args = parser.parse_args()
    athlete = Athlete(
        gamma=args.gamma if args.gamma is not None else 0.85,
        alpha=args.alpha if args.alpha is not None else 0.70,
        x_reduce=args.x_reduce if args.x_reduce is not None else 15,
        y_reduce=args.y_reduce if args.y_reduce is not None else 10,
        v_reduce=args.v_reduce if args.v_reduce is not None else 2
    )
    if args.load:
        athlete.load_model(file_path=args.load)
    if args.load and (args.test or args.play):
        athlete.play(fast=args.test)
    else:
        try:
            athlete.train(episodes=args.epochs if args.epochs is not None else 1000)
        except KeyboardInterrupt:
            print('User stopped training')
        print('Nb Q states : {}'.format(len(athlete.Q)))
        print('Max score : {}'.format(athlete.max))
        if args.save:
            athlete.save_model(file_path=args.save)
        athlete.play(fast=False)
