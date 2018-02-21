from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np
from collections import deque
import random
from keras.models import load_model
from challenge_utils import *  # custom functions
import argparse
import time

# Parser to use arguments
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--train", help="mode to train the network: scratch or load",
                    required=False, default="load")
parser.add_argument("-d", "--display", action='store_false', help="display screen or not")
parser.add_argument("-s", "--startstep", help="step you want to start from when training an existing model",type=int, required=False,
                    default=0)
args=vars(parser.parse_args())

# Define some constants
total_steps = 600000
replay_memory_size = 350000
mini_batch_size = 32
gamma = 0.99
evaluation_period = 10000
nb_epochs = total_steps // evaluation_period
epoch=-1

# Create the network or load it from previous training
if(args['train'] == 'scratch'):
    deepQnet, targetNet = createNetwork()
elif(args['train'] == 'load'):
    deepQnet = load_model('model.h5')
    targetNet = load_model('model.h5')
else:
    raise ValueError("train argument should be either 'scratch' or 'load'")

# Load the game and initialization
game = FlappyBird(graphics="fixed")
p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=True,
        display_screen=args['display'])
# frame_skip = 4 in the paper
list_actions = p.getActionSet() 

p.init()
p.reset_game()
screen_x = process_screen(p.getScreenRGB())
stacked_x = deque([screen_x, screen_x, screen_x, screen_x], maxlen=4)
x = np.stack(stacked_x, axis=-1)
replay_memory = MemoryBuffer(replay_memory_size, screen_x.shape, (1,))

mean_score = np.zeros((nb_epochs))
max_score = np.zeros((nb_epochs))
start = time.time()

for step in range(args['startstep'],total_steps):
    # evaluation
    if(step%evaluation_period == 0 and step>0):
        epoch += 1
        print('[Epoch {:d}/{:d}] {:d} steps done in {:.2f} seconds'.format(epoch+1, total_steps//evaluation_period, evaluation_period, time.time()-start))
        print('Starting evaluation...')
        # Save the network
        deepQnet.save('model.h5')
        nb_games = 100
        if (epoch > 0 and max_score[epoch-1] > 40):
            nb_games = 10
        mean_score[epoch], max_score[epoch] = evaluate(p, nb_games, deepQnet)
        print('Score : {}/{} (mean/max)'.format(mean_score[epoch],max_score[epoch]))
        with open('eval.log','a') as f:
            f.write(str(epoch)+','+str(mean_score[epoch])+','+str(max_score[epoch])+'\n')
        print('Evaluation done. Resume training...')
        start = time.time()
    # action selection
    if np.random.rand() < epsilon(step):
        a = np.random.randint(0,2)
    else:
        a = greedy_action(deepQnet, x)
    # step
    r = clip_reward(p.act(list_actions[a]))
    screen_y = process_screen(p.getScreenRGB())
    replay_memory.append(screen_x, a, r, screen_y, p.game_over())
    # train
    if (step > mini_batch_size and step > 10000): 
        X,A,R,Y,D = replay_memory.minibatch(mini_batch_size)
        QY = targetNet.predict(Y)
        QYmax = QY.max(1).reshape((mini_batch_size,1))
        update = R + gamma * (1-D) * QYmax
        QX = deepQnet.predict(X)
        QX[np.arange(mini_batch_size), A.ravel()] = update.ravel()
        deepQnet.train_on_batch(x=X, y=QX)
    # transfer between deepQnet and targetNet
    if (step > 0 and step % 2500 == 0):
        deepQnet.save('model.h5')
        targetNet = load_model('model.h5') 
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

deepQnet.save('model.h5')
print("Training done!")
