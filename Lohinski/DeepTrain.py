import numpy as np
import argparse
from os import path
from math import floor
from ple.games.flappybird import FlappyBird
from ple import PLE
from random import random

from Train import Athlete

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization

DOWN = 0
UP = 1
ACTIONS = [0, 119]
PATH_TO_MODELS = path.join('.', 'models')


class DeepAthlete:
    def __init__(
            self,
            gamma=0.8,
            load=False
    ):
        self.gamma = gamma
        self.max = 0
        self.print_data = dict()
        self.coach = None
        if not load:
            model = Sequential()

            model.add(BatchNormalization(input_shape=(8,)))

            model.add(Dense(32, kernel_initializer='lecun_uniform'))
            model.add(Activation('relu'))
            model.add(Dropout(rate=0.05))

            model.add(Dense(2, kernel_initializer='zeros'))
            model.add(Activation('tanh'))
            model.compile(optimizer="rmsprop", loss='logcosh')

            self.model = model

    def train(self, episodes=1000, show=False):
        epsilon = 1
        epsilon_decay = 1 / episodes
        jumprate = 0.1
        self.print_data = dict({
            'hits': 0,
            'games_played': 1,
            'below_15': 0,
            'pipes_below_15': 0,
            'ep': 0,
            'pipes': 0,
            'episodes': episodes
        })
        game = FlappyBird()
        env = PLE(game,
                  fps=30,
                  frame_skip=1,
                  num_steps=1,
                  force_fps=True,
                  display_screen=show)
        env.init()
        for _ in range(episodes):
            self.print_data['ep'] += 1
            env.reset_game()
            self.print_data['pipes'] = 0
            while not env.game_over():
                state = game.getGameState()
                S = np.array(list(state.values())).reshape(1, len(state))
                Q = self.model.predict(S)
                if random() < epsilon:
                    if self.coach is None:
                        A = UP if random() < jumprate else DOWN
                    else:
                        A = self.coach.act(game.getGameState())
                else:
                    A = np.argmax(Q[0])
                R = env.act(ACTIONS[A])
                if R == 1.0:
                    self.print_data['pipes'] += 1
                Q[0][A] = R if R < 0 else R + self.gamma * np.max(Q[0])
                self.model.fit(
                    x=S, y=Q,
                    epochs=1,
                    verbose=False
                )
            epsilon -= epsilon_decay
            self.print_status()

    def print_status(self):
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
        if path.isfile(file_path):
            r = input('File {} exists. Overwrite ? (y,[n]) '.format(file_path))
            if r != 'y':
                return
        self.model.save(file_path)

    def load_model(self, file_path):
        self.model = load_model(file_path)

    def act(self, state):
        S = np.array(list(state.values())).reshape(1, len(state))
        Q = self.model.predict(S)
        return np.argmax(Q)

    def play(self, fast=True):
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
        print('- Games < 15 pipes: {}'.format(len(tuple(filter(lambda x: x < 15, pipes)))))
        print('- Played {} games'.format(100))
        print('- Average score: {} pipes'.format(np.round(np.mean(pipes), decimals=1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-l', '--load',
        help='Load a specific neural network and weights from a file',
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
        '-c', '--coach',
        type=str,
        help='Load a state engineered coach for exploration rather than random choices'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='To display screen while training'
    )
    args = parser.parse_args()

    athlete = DeepAthlete(
        gamma=args.gamma if args.gamma else 0.85,
        load=args.load is not None
    )
    if args.load:
        athlete.load_model(file_path=args.load)
    if args.load and (args.test or args.play):
        athlete.play(fast=args.test)
    else:
        if args.coach:
            athlete.coach = Athlete()
            athlete.coach.load_model(file_path=args.coach)
        try:
            athlete.train(
                episodes=1000 if args.epochs is None else args.epochs,
                show=args.show
            )
        except KeyboardInterrupt:
            print('User stopped training')
        if args.save:
            athlete.save_model(file_path=args.save)
        athlete.play(fast=False)
