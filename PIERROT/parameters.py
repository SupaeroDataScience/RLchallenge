TOTAL_STEPS = 300000
REPLAY_MEMORY_SIZE = 1000000
MINI_BATCH_SIZE = 32
GAMMA = 0.99

# timesteps to observe before training
OBSERVE = 10000
# frames over which to anneal epsilon
EXPLORE = 1000000
# starting value of epsilon
INITIAL_EPSILON = 0.1
# final value of epsilon
FINAL_EPSILON = 0.001
# learning rate for nn training
LEARNING_RATE = 1e-4
# number of steps between two weights transfert between epxloration and target network
WEIGHT_TRANSFERT = 2500

DISPLAY_SCREEN = False
IMG_HEIGHT = 80
IMG_WIDTH = 80

DQN_SAVE_FILE = "Outputs/dqn.h5"

# logfile path
LOG_FILE = "Outputs/log.csv"
