import pickle
import numpy as np
from math import floor

model_name = 'simple_training'

with open('models/' + model_name + '.pkl', 'rb') as f:
    Q = pickle.load(f)


def state2coord(game_state):
    gap_top = game_state.get('next_pipe_top_y')
    bird_y = game_state.get('player_y')
    player_vel = game_state.get('player_vel')
    pipe_x = game_state.get('next_pipe_dist_to_player')
    return ','.join([
        str(floor(pipe_x / 15)),
        str(floor((gap_top - bird_y) / 15)),
        str(floor(player_vel))
    ])


def FlappyPolicy(state, _):
    S = state2coord(state)
    if Q.get(S) is None:
        Q[S] = np.array([1, 0])
    return np.argmax(Q[S])
