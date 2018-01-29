from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np
import sys
import pickle
from math import floor
from random import random

UP = 1
DOWN = 0
actions = [0, 119]

Simplify_X = 15
Simplify_Y = 20

episodes = 10000
ep = 0

Q = dict()
gamma = 0.80
alpha = 0.65
epsilon = 1
epsilon_decay = epsilon / episodes
jump_rate = 0.2

hits = 0
games_played = 1


def state2coord(game_state):
    gap_top = game_state.get('next_pipe_top_y')
    bird_y = game_state.get('player_y')
    player_vel = game_state.get('player_vel')
    pipe_x = game_state.get('next_pipe_dist_to_player')
    return ','.join([
        str(floor(pipe_x / Simplify_X)),
        str(floor((gap_top - bird_y) / Simplify_Y)),
        str(floor(player_vel))
    ])


bestQ = dict({
    'Q': dict(),
    'pipes': 0
})
sum_pipes = np.zeros(episodes, dtype=int)
game = FlappyBird()
env = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=True, display_screen=False)
env.init()
try:
    for _ in range(episodes):
        ep += 1
        env.reset_game()
        pipes = 0
        S = state2coord(game.getGameState())
        while not env.game_over():
            if Q.get(S) is None:
                Q[S] = np.array([0, 0])
            A = np.argmax(Q[S])
            r = env.act(actions[A])
            if r == 1.0:
                pipes += 1
            R = (1 if r == 0. else 10) if r != -5. else -100
            S_ = state2coord(game.getGameState())
            if Q.get(S_) is not None:
                Q[S][A] = (1 - alpha) * Q[S][A] + alpha * (10*R + gamma * np.max(Q[S_]))
            S = S_
        epsilon -= epsilon_decay
        if pipes > bestQ.get('pipes'):
            print('Average score since last max: {}'.format(hits / games_played))
            print('Max: {}, on game {}'.format(pipes, ep + 1))
            bestQ['Q'] = dict(Q)
            bestQ['pipes'] = pipes
            hits = pipes
            games_played = 1
        else:
            hits += pipes
            games_played += 1
        if ep % floor(episodes / 10 if episodes < 10000 else episodes / 100) == 0:
            print('Games: {}/{} ({}%)'.format(ep, episodes, (100 * ep/episodes)))
except KeyboardInterrupt:
    print('Stopped at game: {}'.format(ep))
print('Nb Q states : {}'.format(len(Q)))
print('Max score : {}'.format(bestQ.get('pipes')))

try:
    env = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=False, display_screen=True)
    env.init()
    while True:
        env.reset_game()
        while not env.game_over():
            S = state2coord(game.getGameState())
            if Q.get(S) is None:
                Q[S] = np.array([0, 0])
            A = np.argmax(Q[S])
            env.act(actions[A])
except KeyboardInterrupt:
    with open('simple_training.pkl', 'wb+') as f:
        pickle.dump(Q, f, pickle.HIGHEST_PROTOCOL)
    sys.exit()
