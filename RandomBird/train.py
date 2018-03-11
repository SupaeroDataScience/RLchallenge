from ple import PLE
from ple.games.flappybird import FlappyBird
import numpy as np
import matplotlib.pyplot as plt
import time
from keras import optimizers
from keras.models import load_model
from collections import deque
from MemoryBufferClass import MemoryBuffer
import parameters
from network import generate_dqn, process_screen, greedy_action, epsilon, epsilon_greedy_action

# Initialize dqn and dqn_target
dqn = generate_dqn()
adam = optimizers.Adam(lr = parameters.LEARNING_RATE)
dqn.compile(loss = "mean_squared_error", optimizer = adam)
dqn.save(parameters.DQN_SAVE_FILE)
dqn_target = load_model(parameters.DQN_SAVE_FILE)

# Load environment
game = FlappyBird()
p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=False, display_screen = parameters.DISPLAY_SCREEN)
p.init()
p.reset_game()
actions = p.getActionSet()

# Initialize state and replay memory
screen_x = process_screen(p.getScreenRGB())
stacked_x = deque([screen_x, screen_x, screen_x, screen_x], maxlen=4)
x = np.stack(stacked_x, axis=-1)
replay_memory = MemoryBuffer(parameters.REPLAY_MEMORY_SIZE, (parameters.IMG_HEIGHT, parameters.IMG_WIDTH), (1,))
game_number = 1
episode_reward = 0
mean_reward = 0
start_time = time.time()

# Open log file
logFile = open(parameters.LOG_FILE, 'w')
logFile.write("Step,Episode,Mean_Reward,Time \n")

# Deep Q-learning with experience replay
for step in range(parameters.TOTAL_STEPS):
    if ((step%500)==0):
	    duration = time.time() - start_time
	    print("Step number %d, time since starting: %.3f sec" % (step, duration))
	    print("Game number %d " % (game_number))
	    print("Epsilon : {}".format(epsilon(step)))
	    logFile.write("%d,%d,%.3f,%.3f \n" % (step, game_number, mean_reward, duration))
		
    # select action
    a = epsilon_greedy_action(dqn, x, step)
    # get reward
    r = p.act(actions[a])
    episode_reward += r
    screen_y = process_screen(p.getScreenRGB())
    d = p.game_over()
    replay_memory.append(screen_x, a, r, screen_y, d)
    # train
    if step > parameters.MINI_BATCH_SIZE and step % 5 == 0 and step  > parameters.OBSERVE:
        X,A,R,Y,D = replay_memory.minibatch(parameters.MINI_BATCH_SIZE)
        QY = dqn_target.predict(Y)
        QYmax = QY.max(1).reshape((parameters.MINI_BATCH_SIZE,1))
        update = R + parameters.GAMMA * (1-D) * QYmax
        QX = dqn.predict(X)
        QX[np.arange(parameters.MINI_BATCH_SIZE), A.ravel()] = update.ravel()
        dqn.train_on_batch(x=X, y=QX)

    # transfert weights between networks
    if step > 1 and step % parameters.WEIGHT_TRANSFERT == 0:
    	dqn.save(parameters.DQN_SAVE_FILE)
    	dqn_target = load_model(parameters.DQN_SAVE_FILE)

    # prepare next transition
    if d==True:
    	# update mean reward over episodes
        mean_reward += (1.0/game_number)*(episode_reward - mean_reward)
        # restart episode
        episode_reward = 0
        p.reset_game()
        game_number += 1
        screen_x = process_screen(p.getScreenRGB())
        stacked_x = deque([screen_x, screen_x, screen_x, screen_x], maxlen=4)
        x = np.stack(stacked_x, axis=-1)
    else:
        # keep going
        screen_x = screen_y
        stacked_x.append(screen_x)
        x = np.stack(stacked_x, axis=-1)

#End of training, let's save the model
dqn.save(parameters.DQN_SAVE_FILE)
print("Model saved ...")
print("Training done :) !")
