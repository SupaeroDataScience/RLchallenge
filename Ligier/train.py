from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np
from collections import deque
import random 
from keras.models import load_model
from challenge_utils import *  # custom functions
import argparse

# Parser to use arguments
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--train", help="mode to train the network: scratch or load",
                    required=False, default="scratch")
parser.add_argument("-d", "--display", help="display screen or not", required=False, default=True)
parser.add_argument("-e", "--evaldisp", help="display screen during evaluation", required=False,
                    default=False)
args=vars(parser.parse_args())

# Define some constants
total_steps = 10000000
replay_memory_size = 100000#0
mini_batch_size = 32
gamma = 0.99
evaluation_period = 10000
nb_epochs = total_steps // evaluation_period
epoch=-1

def MCeval(network, trials, length, gamma):
    scores = np.zeros((trials))
    for i in range(trials):
        p.reset_game()
        screen_x = process_screen(p.getScreenRGB())
        stacked_x = deque([screen_x, screen_x, screen_x, screen_x], maxlen=4)
        x = np.stack(stacked_x, axis=-1)
        for t in range(length):  # SHOULD change this with a while loop
            a = greedy_action(network, x)
            r = clip_reward(p.act(a))
            screen_y = process_screen(p.getScreenRGB())
            scores[i] = scores[i] + gamma**t * r
            if p.game_over():
                # restart episode
                p.reset_game()
                screen_x = process_screen(p.getScreenRGB())
                stacked_x = deque([screen_x, screen_x, screen_x, screen_x], maxlen=4)
                x = np.stack(stacked_x, axis=-1)
            else:
                # keep going
                screen_x = screen_y
                stacked_x.append(screen_x)
                x = np.stack(stacked_x, axis=-1)
    return np.mean(scores)


# Create the network or load it from previous training
if(args['train'] == 'scratch'):
    deepQnet = createNetwork()
elif(args['train'] == 'load'): 
    deepQnet = load_model('model.h5')
else:
    raise ValueError("train argument should be either 'scratch' or 'load'")

# Load the game and initialization
game = FlappyBird()
p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=True,
        display_screen=True)
p.init()
p.reset_game()
screen_x = process_screen(p.getScreenRGB())
stacked_x = deque([screen_x, screen_x, screen_x, screen_x], maxlen=4)
x = np.stack(stacked_x, axis=-1)
replay_memory = MemoryBuffer(replay_memory_size, screen_x.shape, (1,))

Xtest = np.array([x])
scoreQ = np.zeros((nb_epochs))
scoreMC = np.zeros((nb_epochs))

for step in range(total_steps):
    # evaluation
    if(step%evaluation_period == 0 and step>0): #
        deepQnet.save('model.h5')
        epoch += 1
        # evaluation of initial state
        scoreQ[epoch] = np.mean(deepQnet.predict(Xtest).max(1))
        # roll-out evaluation
        scoreMC[epoch] = MCeval(network=deepQnet, trials=20, length=700, gamma=gamma)
        print('Score at step {}: {}'.format(step,scoreMC[epoch]))
    # action selection
    if np.random.rand() < epsilon(step):
        a = random.choice([119, 0])  #None
    else:
        a = greedy_action(deepQnet, x)
    # step
    r = p.act(a)
    screen_y = process_screen(p.getScreenRGB())
    replay_memory.append(screen_x, a, r, screen_y, p.game_over())
    # train
    if step>mini_batch_size:
        X,A,R,Y,D = replay_memory.minibatch(mini_batch_size)
        QY = deepQnet.predict(Y)
        QYmax = QY.max(1).reshape((mini_batch_size,1))
        update = R + gamma * (1-D) * QYmax
        QX = deepQnet.predict(X)
        A[A == 119] = 1
        QX[np.arange(mini_batch_size), A.ravel()] = update.ravel()
        deepQnet.train_on_batch(x=X, y=QX)
    # prepare next transition
    if p.game_over()==True:
        # restart episode
        p.reset_game()
        screen_x = process_screen(p.getScreenRGB())
        stacked_x = deque([screen_x, screen_x, screen_x, screen_x], maxlen=4)
        x = np.stack(stacked_x, axis=-1)
    else:
        # keep going
        screen_x = screen_y
        stacked_x.append(screen_x)
        x = np.stack(stacked_x, axis=-1)

