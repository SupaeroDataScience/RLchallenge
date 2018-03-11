import numpy as np

Q = np.load("Q_matrix_QLEARNING_BEST.npy")

actions = [None, 119]

state_min = np.array([0,
                      -48,
                      -10])

bin_c = np.array([
    4,
    8,
    2])

state_shape = np.array([round((284)/4+1),
                        round(704/8+1),
                        round(40/2+1)])

print(f"The tabular Q contains {sum(Q.shape):_} value-action values.")


def FlappyPolicy(state, screen):

	# Discretize state
    s = get_engineered_state(state)

    # Ravel indexes
    s = np.ravel_multi_index(s, state_shape)

    # Greedy policy
    a = Q[s, :].argmax()

    # Return greedy action
    return actions[a]


def get_engineered_state(state):
    y = state['player_y']
    speed = state['player_vel']
    next_y_b = state['next_pipe_bottom_y']
    next_dist = state['next_pipe_dist_to_player']

    engineered_state = np.array([next_dist,
                                 next_y_b - y,
                                 speed])

    s = np.round(engineered_state/bin_c - state_min).astype(int)

    return s
