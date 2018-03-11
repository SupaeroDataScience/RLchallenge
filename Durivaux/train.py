import sys
import os

from ple.games.flappybird import FlappyBird
from ple import PLE

import numpy as np

from collections import deque
from time import time

from MemoryBuffer import MemoryBuffer

from skimage import transform, color

from keras.models import Sequential, load_model
from keras.layers import Conv2D, Flatten, Dense
from keras.optimizers import Adam
from keras import backend as kerasBackend

# Display screen or not
DISPLAY = True

# Print period (in steps) of general info
PRINT_PERIOD = 5000 # %1000=0

### Constants
GAMMA = 0.99
STEPS_NB = 1000000
REPLAY_MEM_SIZE = STEPS_NB
MINIBATCH_SIZE = 32
STEPS_BETWEEN_TRAINING = 5
# Weight transfer (exploration -> target network)
STEPS_BETWEEN_TRANSFER = 2500
# Epsilon
INITIAL_EXPLORATION = 20000
INITIAL_EPSILON = 0.1
FINAL_EPSILON = 1e-3
EXPLORATION_STEPS = 500000
# DQN learning rate
LEARNING_RATE = 1e-4
# Evaluation period (in steps)
EVALUATION_PERIOD = 25000
# Backup period (in steps) of the dqn
BACKUP_PERIOD = 25000
# Number of steps during which the action is repeated (1st: choice, then repeat)
REPEAT = 2 # human-like is max 10 Hz (min 3)


def processScreen(screen):
    """
    Resize and gray-ify screen
    """
    return 255*transform.resize(color.rgb2gray(screen[60:,25:310,:]),(80,80))


def epsilon(step):
    """
    Epsilon for exploration/exploitation trade-off
    """
    if step < INITIAL_EXPLORATION:
        return 1
    elif step < EXPLORATION_STEPS:
        return INITIAL_EPSILON + (FINAL_EPSILON - INITIAL_EPSILON)/(EXPLORATION_STEPS-INITIAL_EXPLORATION) * (step-INITIAL_EXPLORATION)
    else:
        return FINAL_EPSILON


def epsilonGreedy(dqn, x, step):
    """
    Epsilon-greedy action
    """
    if np.random.rand() < epsilon(step):
        return np.random.randint(2)
    else:
        return np.argmax(dqn.predict(np.array([x])))


def initX(rgb):
    """
    Initialize screenX and x
    """
    _screenX = processScreen(rgb)
    _stackedX = deque([_screenX]*4, maxlen=4)
    _x = np.stack(_stackedX, axis=-1)
    return _screenX, _stackedX, _x


def createDQN():
    """
    Create deep Q network
    """
    # Cf DeepMind article
    dqn = Sequential()
    dqn.add(Conv2D(filters=16, kernel_size=(8,8), strides=4, activation='relu', input_shape=(80,80,4)))
    dqn.add(Conv2D(filters=32, kernel_size=(4,4), strides=2, activation='relu'))
    dqn.add(Flatten())
    dqn.add(Dense(units=256, activation='relu'))
    dqn.add(Dense(units=2, activation='linear'))
    
    dqn.compile(optimizer=Adam(lr=LEARNING_RATE), loss='mean_squared_error')
    return dqn


def evaluate(p, nbOfGames, dqn):
    """
    Evaluation
    """
    actions = p.getActionSet()
    scores = np.zeros((nbOfGames))
    for i in range(nbOfGames):
        frames = deque([np.zeros((80,80))]*4, maxlen=4)
        p.reset_game()
        frame = 0
        while not p.game_over():
            screen = processScreen(p.getScreenRGB())
            frames.append(screen)
            x = np.stack(frames, axis=-1)
            if frame % REPEAT == 0 and REPEAT != 1 and frame > 0:
                a = lastAction
            else:
                a = actions[np.argmax(dqn.predict(np.expand_dims(x, axis=0)))]
                lastAction = a
            r = p.act(a)
            scores[i] += r
            frame += 1
    return np.min(scores), np.mean(scores), np.max(scores)


### MAIN

# Try to load DQN, or create a new one
loaded = False
if len(sys.argv) == 2 and sys.argv[1] == "load":
    try:
        dqnExploration = load_model('dqn.h5')
        loaded = True
        INITIAL_EXPLORATION = 0
        print("Loaded 'dqn.h5'")
    except:
        print("Usage: train.py [load]")
        print("Defaulting to new network")
        dqnExploration = createDQN()
else:
    print("Starting from scratch")
    dqnExploration = createDQN()
dqnExploration.save('dqn.h5')
dqnTarget = load_model('dqn.h5')

# Environment
game = FlappyBird()
p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=True, display_screen=DISPLAY)
actions = p.getActionSet()

# Initialization
gameNumber = 1
p.init()
p.reset_game()
meanScore = -5

screenX, stackedX, x = initX(p.getScreenRGB())
replayMemory = MemoryBuffer(REPLAY_MEM_SIZE, (80,80), (1,))

start = time()

# Q-learning
if not loaded:
    print("START OF INITIAL EXPLORATION")
for step in range(STEPS_NB):
    # End of initial exploration
    if step == INITIAL_EXPLORATION and not loaded:
        print("END OF INITIAL EXPLORATION")

    # Final epsilon
    if step == EXPLORATION_STEPS:
        print("FINAL EPSILON REACHED")

    # Print progress info
    if step % PRINT_PERIOD == 0 and step > 0:
        print("Step {}k\tGame number: {}\tElapsed time: {:.0f} s\tEpsilon: {:.3f}".format(step//1000, gameNumber, time()-start, epsilon(step)))

    # Backup dqn and clear Keras session
    if step % BACKUP_PERIOD == 0 and step > 0:
        backupName = 'dqn-backup-'+str(step//1000)+'k.h5'
        dqnExploration.save(backupName)
        dqnTarget.save('dqnTarget.h5')
        
        # in order not to get clogged up
        stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        kerasBackend.clear_session()
        sys.stdout = stdout

        dqnExploration = load_model(backupName)
        dqnTarget = load_model('dqnTarget.h5')
        print("*** Saved dqn backup")

    # Evaluation
    if step % EVALUATION_PERIOD == 0 and step > 0 and step > INITIAL_EXPLORATION:
        print("Evaluation ", end="")
        sys.stdout.flush()
        numberOfGames = 100
        if 40 <= meanScore < 80:
            numberOfGames = 50
        elif 80 <= meanScore:
            numberOfGames = 25
        minScore, meanScore, maxScore = evaluate(p, numberOfGames, dqnExploration)
        print("---> Min: {:.0f}\tMean: {:.1f}\tMax: {:.0f}".format(minScore, meanScore, maxScore))

    sys.stdout.flush()
    
    # Act (epsilon-greedy)
    if step % REPEAT == 0 and step > 0:
        a = lastAction
    else:
        a = epsilonGreedy(dqnExploration, x, step)
        lastAction = a
    r = p.act(actions[a])
    d = p.game_over()

    # Reward clipping
    if r != -1 and r != 1:
        r = 0.1 # keep the bird moving on?

    # Next screen
    screenY = processScreen(p.getScreenRGB())
    replayMemory.append(screenX, a, r, screenY, d)

    # Train (minibatch)
    if step > INITIAL_EXPLORATION and step > MINIBATCH_SIZE and step % STEPS_BETWEEN_TRAINING == 0:
        X, A, R, Y, D = replayMemory.minibatch(MINIBATCH_SIZE)
        QY = dqnTarget.predict(Y)
        QYmax = QY.max(1).reshape((MINIBATCH_SIZE, 1))
        update = R + GAMMA * (1-D) * QYmax
        QX = dqnExploration.predict(X)
        QX[np.arange(MINIBATCH_SIZE), A.ravel()] = update.ravel()
        dqnExploration.train_on_batch(x=X, y=QX)

    # Update target dqn
    if step % STEPS_BETWEEN_TRANSFER == 0:
        dqnExploration.save('dqn.h5')
        dqnTarget = load_model('dqn.h5')

    # Prepare next step
    if p.game_over():
        gameNumber += 1
        p.reset_game()
        screenX, stackedX, x = initX(p.getScreenRGB())
    else:
        screenX = screenY
        stackedX.append(screenX)
        x = np.stack(stackedX, axis=-1)

print("Final evaluation ", end="")
sys.stdout.flush()
numberOfGames = 100
minScore, meanScore, maxScore = evaluate(p, numberOfGames, dqnExploration)
print("---> Min: {:.0f}\tMean: {:.1f}\tMax: {:.0f}".format(minScore, meanScore, maxScore))
sys.stdout.flush()

dqnExploration.save('dqn.h5')
print('Done')
