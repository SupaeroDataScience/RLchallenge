# A file where we store all of our constants

# The different actions we can perform for our game
ACTIONS = [None, 119]

# The different states of the game
#   1 - Flappy position and velocity
#   2 - Distance to pipes
#   3 - Next pipe info
#   4 - Next next pipe info
STATES = [
    'player_y', 'player_vel',
    'next_pipe_dist_to_player', 'next_next_pipe_dist_to_player',
    'next_pipe_top_y', 'next_pipe_bottom_y',
    'next_next_pipe_top_y', 'next_next_pipe_bottom_y',
]

# DQN parameters
# Set the size of the image we want to deal with
IMG_SHAPE = (80, 80)
ACTION_SHAPE = (1,)

# The number of frames we need in the stack
STACK_SIZE = 4

# Input shape
INPUT_SHAPE = IMG_SHAPE + (STACK_SIZE,)

# Learning rate for the optimizer (default: Adam)
LEARNING_RATE = 1e-4
