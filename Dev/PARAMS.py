""" Constants and parameters used in train.py and run.py """

# constants
LIST_ACTIONS = [0, 119]         # Possible actions. No jump (0) or jump (119)
SIZE_IMG = (80, 80)             # Size of the processed image

# Parameters
LEARNING_RATE = 1e-4            # Learning rate

DISPLAY_GAME = False            # False to train, True to visualize Flappy
TOTAL_STEPS = 70000            # nb of frames
STEPS_TARGET = 2500             # Changing target dqn
GAMMA = 0.99

INITIALIZATION = 200          # we fill the buffer with totally random frames during n steps
EXPLORATION_STEPS = 1e6         # epsilon slowly decreases during EXPLORATION_STEPS steps
EPSILON0 = 0.1                  # Initial epsilon
FINAL_EPSILON = 1e-3            # Final epsilon

EVALUATION = True               # True to evaluate. False otherwise (faster).
EVALUATION_PERIOD = 250       # Evaluation period. Run an evaluation every step period

REPLAY_MEMORY_SIZE = 70000     # replay_memory_size
MINI_BATCH_SIZE = 32            # Size of the mini batch
