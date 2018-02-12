from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
from collections import deque
from matplotlib import pyplot as plt
from time import time
import sys
from CNN import CNN
from BufferRL4 import MemoryBuffer
from PARAMS import *

def select_action(model, x):
    neural_value = model.predict(np.array([x]))
    #print(neural_value)
    print("Neural values are {}".format(neural_value))
    return np.argmax(neural_value)


def epsilon(step):
    if step < 10000:
        return 0.8 - step*7.99e-5
    return .001


def process_screen(screen):
    screen_cut = screen[50:-1, 0:400] # cut
    screen_grey = 256 * (rgb2gray(screen_cut)) # in gray
    output = resize(screen_grey, (84, 84), mode='constant') # resize
    return output


def CNN_generation(directory, continue_training):
    path_model = "model_dql_flappy3_dense.h5"
    path_buffer = "buffer_flappy3_dense.pkl"
    path_step = "step_flappy3_dense.npy"
    path_score = "score_flappy3_dense.npy"

    cnn = CNN(directory, path_model, path_buffer, path_step, path_score)
    if continue_training:
        cnn.load()
    else:
        cnn.init()
    return cnn


def clip_reward(r, dead):
    # dead = false when alive / true otherwise
    if not dead:
        rr = 0.1
    else:
        rr = -1

    if not dead and r > 0: # pass a pipe and still alive
        rr *= 10

    return rr


def MCeval(network, trials, length, gamma):
    scores = np.zeros((trials))
    for i in range(trials):
        p.reset_game()
        screen = p.getScreenRGB()
        screen_x = process_screen(screen)
        stacked_x = deque([screen_x, screen_x, screen_x, screen_x], maxlen=4)
        x = np.stack(stacked_x, axis=-1)
        #print(x.shape)
        for t in range(length):
            q = select_action(network, x)
            a = list_actions[q]
            raw_screen_y, r, d = game_step(p,a)
            r = clip_reward(r, d)
            screen_y = process_screen(raw_screen_y)
            scores[i] = scores[i] + gamma**t * r
            if d==True:
                # restart episode
                p.reset_game()
                screen = p.getScreenRGB()
                screen_x = process_screen(screen)
                stacked_x = deque([screen_x, screen_x, screen_x, screen_x], maxlen=4)
                x = np.stack(stacked_x, axis=-1)
            else:
                # keep going
                screen_x = screen_y
                stacked_x.append(screen_x)
                x = np.stack(stacked_x, axis=-1)
    return np.mean(scores)


def game_step(p, a):
    reward = p.act(a)
    raw_screen_y = p.getScreenRGB()
    d = p.game_over()

    return raw_screen_y, reward, d

if __name__ == "__main__":

    # Init game
    list_actions = [0, 119]
    game = FlappyBird()
    if DISPLAY_GAME:
        p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=True, display_screen=True)
    else:
        p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=True, display_screen='store_false')

    p.init()
    reward = 0.0

    # Generate CNN
    cnn = CNN_generation("Save/", CONTINUE_TRAINING)
    cnn_target = cnn

    # Training
    p.reset_game()
    screen = p.getScreenRGB()
    screen_x = process_screen(screen)
    stacked_x = deque([screen_x, screen_x, screen_x, screen_x], maxlen=4)
    x = np.stack(stacked_x, axis=-1)
    if CONTINUE_TRAINING:
        replay_memory = cnn.buffer
    else:
        replay_memory = MemoryBuffer(REPLAY_MEMORY_SIZE, (84, 84), (1,))

    # initial state for evaluation
    Xtest = np.array([x])
    nb_epochs = TOTAL_STEPS // EVALUATION_PERIOD
    score = np.zeros((2, nb_epochs))    # [scoreQ, scoreMC]

    # Deep Q-learning with experience replay
    for step in range(cnn.step+1, TOTAL_STEPS):
        print("Step {} / {}".format(step, TOTAL_STEPS))
        # t1 = time()
        # evaluation every EVALUATION_PERIOD
        if step % EVALUATION_PERIOD == 0 and step > 0:
            print("Evaluating...")
            epoch = step // EVALUATION_PERIOD
            # evaluation of initial state
            score[0, epoch] = np.mean(cnn.model.predict(Xtest).max(1))
            # roll-out evaluation
            score[1, epoch] = MCeval(network=cnn.model, trials=20, length=700, gamma=GAMMA)
            print('Score: ', score[:, epoch])

        # action selection
        print("Epsilon : ", epsilon(step))
        if step < INITIALIZATION:
            a = 119*np.random.randint(0, 2)*np.random.randint(0, 2)
        else:
            if np.random.rand() < epsilon(step):
                a = 119*np.random.randint(0, 2)*np.random.randint(0, 2)
            else:
                q = select_action(cnn.model, x)
                a = list_actions[q]
                print('Action made: ', a)


        # step
        raw_screen_y, r, d = game_step(p, a)
        r = clip_reward(r, d)
        screen_y = process_screen(raw_screen_y)
        replay_memory.append(screen_x, a, r, screen_y, d)

        # train
        if step > max(MINI_BATCH_SIZE, INITIALIZATION):
            X, A, R, Y, D = replay_memory.minibatch(mini_batch_size)
            QY = cnn_target.model.predict(Y)
            QYmax = QY.max(1).reshape((MINI_BATCH_SIZE, 1))
            update = R + GAMMA * (1 - D) * QYmax
            QX = cnn.model.predict(X)
            QX[np.arange(MINI_BATCH_SIZE), 0] = update.ravel()
            cnn.model.train_on_batch(x=X, y=QX)

        # modification of cnn_target
        if step % 3000 and step > 0:
            cnn_target = cnn

        # save every steps_to_save
        if step % STEPS_TO_SAVE == 0 and step > 0 and PARTIAL_SAVE:
            cnn.buffer = replay_memory
            cnn.step = step
            cnn.score = score
            cnn.save_all()

        # prepare next transition
        if d:
            # restart episode
            p.reset_game()                                          # RESET game
            screen_x = process_screen(p.getScreenRGB())             # Next frame = first frame
            stacked_x = deque([screen_x, screen_x, screen_x, screen_x], maxlen=4)  #
            x = np.stack(stacked_x, axis=-1)  # x <- image stacked
        else:
            # keep going
            screen_x = screen_y                                     # Next frame = next frame
            stacked_x.append(screen_x)
            x = np.stack(stacked_x, axis=-1)

        # t2 = time()
        # print("Time (s) spent on this step : ", t2-t1)
        print("")

    # Save CNN
    cnn.save_cnn()