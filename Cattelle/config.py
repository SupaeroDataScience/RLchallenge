class Config:
    # Experience replay settings
    ER_SIZE = 20000  # Total number of samples to keep in the Experience Replay memory
    HISTORY_LENGTH = 4  # Number of frames to keep in each state
    MINIBATCH_SIZE = 32  # Number of samples in a single minibatch, must be < MIN_ER_SIZE

    # Network settings
    OPTIMISER = 'rmsprop'
    LEARNING_RATE = 1e-6
    DECAY = 0.9
    MOMENTUM = 0.95

    # Learning settings
    INITIALISATION_STD = 0.1  # Standard deviation used for initialising weights of the conv2d layers
    TIMESTEPS = 100000  # Number of timesteps used for the learning, one action is taken during one step.
    INITIAL_EPS = 1.0  # Initial value for the exploratory parametr epsilon
    DISCOUNT_RATE = 0.95  # Parameter for the gamma discount rate
    MIN_ER_SIZE = 3000  # Minimum number of samples in the ER to begin learning
    TEST_DELTA = 10000  # Number of timesteps between two successive tests of the network
    NUM_TEST_TRIALS = 10  # Number of trials to conduct during each test session
    PROB_FLAP = 1 / 4  # Probability of action "flap" (119) when taking a random action during the exploration

    # Simulator settings
    REWARD_ALIVE = 0.1  # Reward granted for each timestep where the player remains alive (except if it passes a pipe)

    # Misc. settings
    MODEL_FILENAME = 'dqn.h5'
    SAVE_DELTA = 5000  # Number of timesteps between two successive saves of the network, must be > MIN_ER_SIZE


class DebugConfig(Config):
    TIMESTEPS = 100
    MIN_ER_SIZE = 10
    MINIBATCH_SIZE = 5
    ER_SIZE = 50
    SAVE_DELTA = 50
    TEST_DELTA = 25
