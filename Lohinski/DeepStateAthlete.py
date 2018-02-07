import sys
import numpy as np
from os import path
from math import floor
from ple.games.flappybird import FlappyBird
from ple import PLE
from random import random

from StateEngineeringAthlete import Athlete

from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization

DOWN = 0
UP = 1
ACTIONS = [0, 119]
PATH_TO_MODELS = path.join('.', '.models')


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

            # model.add(BatchNormalization(input_shape=(8,)))

            model.add(Dense(64, kernel_initializer='lecun_uniform', input_shape=(8, )))
            model.add(Activation('relu'))

            model.add(Dense(32, kernel_initializer='lecun_uniform'))
            model.add(Activation('relu'))
            model.add(Dropout(rate=0.2))

            model.add(Dense(2, kernel_initializer='zeros'))
            model.add(Activation('linear'))
            model.compile(loss='mean_squared_error', optimizer="rmsprop")

            self.model = model

    def train(self, episodes=1000, show=False):
        epsilon = 1
        epsilon_decay = 1 / episodes
        jumprate = 0.1
        buffer = []
        bufferSize = 10
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
            state = game.getGameState()
            S = np.array(list(state.values())).reshape(1, len(state))
            while not env.game_over():
                Q = self.model.predict(S)
                if random() < epsilon:
                    if self.coach is None:
                        A = UP if random() < jumprate else DOWN
                    else:
                        A = self.coach.act(game.getGameState())
                else:
                    A = np.argmax(Q[0])
                r = env.act(ACTIONS[A])
                if r == 1.0:
                    self.print_data['pipes'] += 1
                R = self.biase_reward(r)
                state_ = game.getGameState()
                S_ = np.array(list(state_.values())).reshape(1, len(state))
                buffer.append((S, A, R, S_))
                if len(buffer) > bufferSize:
                    buffer.pop(0)
                    X_train = []
                    Y_train = []
                    for m in buffer:
                        S, A, R, S_ = m
                        Q = self.model.predict(S)
                        Q_ = self.model.predict(S_)
                        y = np.zeros((1, 2))
                        y[:] = Q[:]
                        if R < 0:
                            y[0][A] = R
                        else:
                            y[0][A] = R + self.gamma * np.max(Q_[0])
                        X_train.append(S.reshape(len(state),))
                        Y_train.append(np.array(y).reshape(2,))
                    X_train = np.array(X_train)
                    Y_train = np.array(Y_train)
                    self.model.fit(
                        x=X_train, y=Y_train,
                        batch_size=bufferSize,
                        epochs=1,
                        verbose=False
                    )
                S = S_
            epsilon -= epsilon_decay
            self.print_status()

    def biase_reward(self, r):
        # return (1 if r == 0. else 5) if r != -5. else 0
        # return (1 if r == 0. else 2) if r != -5. else -1
        return floor(r)

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

    def save_model(self, name=None):
        if name is None:
            name = 'DeepTrainingModel.pkl'
        file_path = path.join(PATH_TO_MODELS, name)
        if path.isfile(file_path):
            r = input('File {} already exists. Overwrite ? (y,[n]) '.format(name))
            if r != 'y':
                name = 'bis_{}'.format(name)
                self.save_model(name=name)
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
    athlete = DeepAthlete(load=False)
    epochs = 10000
    no_train = False
    test_only = False
    save_name = None
    show = False
    load = False
    if len(sys.argv) > 1:
        try:
            args = sys.argv[1:]
            commands = filter(lambda x: '--' in x, args)
            if any([x not in [
                '--load', '--epochs', '--play', '--save', '--test', '--coach', '--show'
            ] for x in commands]):
                raise IndexError
            if '--load' in args:
                index = args.index('--load')
                path_to_file = args[index + 1]
                athlete = DeepAthlete(load=True)
                athlete.load_model(file_path=path_to_file)
                load = True
                if '--play' in args:
                    no_train = True
                if '--test' in args:
                    no_train = True
                    test_only = True
            if '--show' in args:
                show = True
            if '--save' in args:
                index = args.index('--save')
                save_name = args[index + 1]
            if '--epochs' in args:
                index = args.index('--epochs')
                epochs = int(args[index + 1])
            if '--coach' in args:
                index = args.index('--coach')
                file_path = args[index + 1]
                coach = Athlete()
                coach.load_model(file_path=file_path)
                athlete.coach = coach
            if '--load' not in args and ('--play' in args or '--test' in args):
                print('Can not play/test without loading a model.')
                raise IndexError
        except IndexError:
            print('Error. Please give the correct inputs as described below:')
            print('--load <path_to_model.pkl>,\n'
                  '--epochs <nb_of_epochs>,\n'
                  '--save <name_of_save_file.pkl>,\n'
                  '--play,\n'
                  '--coach <path_to_model.pkl>'.format(__file__))
            sys.exit(1)
    if not no_train:
        try:
            athlete.train(episodes=epochs, show=show)
        except KeyboardInterrupt:
            print('User stopped training')
        print('Max score : {}'.format(athlete.max))
        athlete.save_model(name=save_name)
    athlete.play(fast=test_only)
