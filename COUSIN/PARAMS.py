# Parameters

DISPLAY_GAME = True         # False to train, True to visualize Flappy
CONTINUE_TRAINING = False   # False if Reset of the cnn, True to load one to continue the training
TOTAL_STEPS = 20000  # nb of frames
GAMMA = 0.99

EVALUATION_PERIOD = 25000

REPLAY_MEMORY_SIZE= 20000  # keep it all ?replay_memory_size
MINI_BATCH_SIZE = 32
INITIALIZATION = 2000  # we fill the buffer with totally random frames

PARTIAL_SAVE = False
STEPS_TO_SAVE = 30000  # Everything saved every steps_to_save