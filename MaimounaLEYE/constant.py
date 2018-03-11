#All constants parameters are defined here

# The different actions we can perform for our game, 119 corresponds to the bird 
#going upwards and zero or none correpospond to the bird going downwards
ACTIONS = [None, 119]

# Constant "parameters" of our DeepQNetwork

#Definition of the size of the image we want to operate
IMG_SHAPE = (80, 80)
ACTION_SHAPE = (1,)

# The number of frames needed in the stack
NB_FRAMES = 4

# Input shape
INPUT_SHAPE = IMG_SHAPE + (NB_FRAMES,)

# Learning rate for the optimizer 
LEARNING_RATE = 1e-4
