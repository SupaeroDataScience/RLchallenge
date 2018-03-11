# A file where we store all of our constants and parameters

# The different actions we can perform for flappy bird (flap (119) or not (None))
ACTIONS = [None, 119]

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
