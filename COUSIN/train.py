from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
import sys
from collections import deque
from matplotlib import pyplot as plt

from CNN import CNN
from BufferRL4 import MemoryBuffer

def select_action(model, screen):
    neural_value = model.predict(screen)
    if round(neural_value[0,0]) == 1:
        return 119
    else:
        return None

def epsilon(step):
    if step<1e6:
        return 1.-step*9e-7
    return .1

def process_screen(screen):
    screen_cut = screen[50:-1, 0:400] # cut
    screen_grey = 256 * (rgb2gray(screen_cut)) # in gray
    output = resize(screen_grey, (84, 84), mode='constant') # resize
    return output

def model_CNN_generation(path_CNN, continue_training):
    cnn = CNN()
    if continue_training:
        try:
            print("Loading an existing CNN...")
            cnn.load(path_CNN)
        except IOError:
            print("File not found : ", path_CNN)
            sys.exit()
    else:
        print("Creating a new CNN...")
        cnn.init()
        return cnn.model

def clip_reward(r):
    rr=0
    if r>0:
        rr=1
    if r<0:
        rr=-1
    return rr

def MCeval(network, trials, length, gamma):
    scores = np.zeros((trials))
    for i in range(trials):
        p.reset_game()
        screen = p.getScreenRGB()
        screen_x = process_screen(screen)
        stacked_x = deque([screen_x, screen_x, screen_x, screen_x], maxlen=4)
        x = np.stack(stacked_x, axis=-1)
        print(x.shape)
        for t in range(length):
            a = select_action(network, np.array([x]))
            raw_screen_y, r, d = game_step(p,a)
            r = clip_reward(r)
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

    # Parameters
    continue_training = False
    path_CNN = "model_dql_flappy3_dense.dqf"
    total_steps = 1500
    replay_memory_size = 500
    mini_batch_size = 32
    gamma = 0.99
    evaluation_period = 10


    # Init game
    game = FlappyBird()
    p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=True, display_screen=True)
    p.init()
    reward = 0.0

    # Generate CNN
    model = model_CNN_generation(path_CNN, continue_training)


    # Training
    print("Initialization of the training...")
    p.reset_game()
    screen = p.getScreenRGB()
    screen_x = process_screen(screen)
    stacked_x = deque([screen_x, screen_x, screen_x, screen_x], maxlen=4)
    x = np.stack(stacked_x, axis=-1)
    replay_memory = MemoryBuffer(replay_memory_size, (84, 84), (1,))
    # initial state for evaluation
    Xtest = np.array([x])

    nb_epochs = total_steps // evaluation_period
    epoch = -1
    scoreQ = np.zeros((nb_epochs))
    scoreMC = np.zeros((nb_epochs))

    # Deep Q-learning with experience replay
    print("Computing training...")
    for step in range(total_steps):
        print("Step : ", step)
        # restart episode
        p.reset_game()
        screen = p.getScreenRGB()
        screen_x = process_screen(screen)
        stacked_x = deque([screen_x, screen_x, screen_x, screen_x], maxlen=4)  #
        x = np.stack(stacked_x, axis=-1)  # x <- image stacked
        game_started = False

        while (not p.game_over()):
            if game_started :
                #game continue
                screen_x = screen_y  # x <- image suivante stacked
                stacked_x.append(screen_x)
                x = np.stack(stacked_x, axis=-1)
            else:
                game_started = True

            # evaluation
            if (step+1 % evaluation_period == 0):
                epoch = epoch + 1
                print("epoch = ", epoch)
                # evaluation of initial state
                #print("Before prediction")
                scoreQ[epoch] = round(model.predict(Xtest)[0,0])*119
                #print("End prediction : ", scoreQ[epoch])
                # roll-out evaluation
                scoreMC[epoch] = MCeval(network=model, trials=2, length=7, gamma=gamma)
                #print("Evaluation : ", scoreMC[epoch])

            # action selection
            if np.random.rand() < epsilon(step):
                a = 119 * np.random.randint(0,2)
            else:
                a = select_action(model, np.array([x]))

            # step
            raw_screen_y, r, d = game_step(p, a)
            r = clip_reward(r)
            screen_y = process_screen(raw_screen_y)
            replay_memory.append(screen_x, a, r, screen_y, d)

            # train
            if step > mini_batch_size:
                X, A, R, Y, D = replay_memory.minibatch(mini_batch_size)
                QY = model.predict(Y)
                QYmax = QY.max(1).reshape((mini_batch_size, 1))
                update = R + gamma * (1 - D) * QYmax
                QX = model.predict(X)
                QX[np.arange(mini_batch_size), 0] = update.ravel()
                model.train_on_batch(x=X, y=QX)




    # Save CNN
    print("Saving the new CNN...")
    model.save(path_CNN)