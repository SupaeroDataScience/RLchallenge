from keras.models import load_model
from ple import PLE
from ple.games.flappybird import FlappyBird
import os
import numpy as np
from COUSIN.nn_Tools import *
import COUSIN.nn_PARAMS as params
import sys

if __name__ == "__main__":
    # paths files
    path_model = "Save/model_dqn_flappy3.h5"
    logfile = "Save/logfile.txt"

    # Delete previous logfile
    if os.path.isfile(logfile):
        os.remove(logfile)
    else:
        print("No logfile found !")

    # Init game
    game = FlappyBird(graphics="fixed")
    # TODO: considering to change frame_skip ?
    # frame_skip=4 for some atari games, 3 for others.... To change?
    if params.DISPLAY_GAME:
        p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=True, display_screen=True)
    else:
        p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=True, display_screen='store_false')

    p.init()
    p.reset_game()

    # Training
    dict_x = game.getGameState()
    state_x = list(dict_x.values())
    replay_memory = MemoryBuffer(params.REPLAY_MEMORY_SIZE, params.SIZE_STATE, (1,))

    # Generate NN
    nn = create_nn()
    nn.save(filepath=path_model)
    nn_target = load_model(filepath=path_model)

    # Evaluation barrier
    mean_score = 0

    # Q-learning with experience replay
    for step in range(params.TOTAL_STEPS):
        print("Step {} / {} ----> epsilon={}".format(step, params.TOTAL_STEPS, epsilon(step)))

        if step % params.EVALUATION_PERIOD == 0 and step > 0 and params.EVALUATION and mean_score < 80:
            print("Evaluating...")
            epoch = step // params.EVALUATION_PERIOD
            mean_score, max_score, min_score = evaluation(p, network=dqn, epoch=epoch, trials=20, logfile=logfile)
            print('Score min/max ({}/{}) and mean ({})'.format(min_score, max_score, mean_score))

        # action selection
        if np.random.rand() < epsilon(step):
            a = np.random.randint(0, 2)
        else:
            nn.summary()
            a = greedy_action(nn, state_x)     # 0 or 1

        # step
        r = p.act(params.LIST_ACTIONS[a])
        rr = clip_reward(r)

        dict_y = game.getGameState()
        state_y = list(dict_y.values())
        d = p.game_over()
        replay_memory.append(state_x, a, rr, state_y, d)

        # print some info
        if d:
            print("##################################################################################################")
            print("#################################################################### DEAD ########################")
            print("##################################################################################################")
        if r > 0:
            print("--------------------------------------------------------------------------------------------------")
            print("-------------------------------------------------------------------- Pipe passed ! ---------------")
            print("--------------------------------------------------------------------------------------------------")

        # train
        if step > max(params.MINI_BATCH_SIZE, params.INITIALIZATION):
            X, A, R, Y, D = replay_memory.minibatch(params.MINI_BATCH_SIZE)
            QY = nn_target.predict(Y)
            QYmax = QY.max(1).reshape((params.MINI_BATCH_SIZE, 1))
            update = R + params.GAMMA * (1 - D) * QYmax
            QX = nn.predict(X)
            QX[np.arange(params.MINI_BATCH_SIZE), A.ravel()] = update.ravel()
            nn.train_on_batch(x=X, y=QX)

        # modification of dqn_target
        if step % params.STEPS_TARGET == 0 and step > params.INITIALIZATION:
            nn.save(filepath=path_model)
            nn_target = load_model(filepath=path_model)

        # prepare next transition
        if d:
            # restart episode
            p.reset_game()                                          # RESET game
            dict_x = game.getGameState()
            state_x = list(dict_x.values())
        else:
            # keep going
            dict_x = dict_y
            state_x = list(dict_x.values())                                     # Next frame = next frame
            #stacked_x.append(screen_x)
            #x = np.stack(stacked_x, axis=-1)


    # Save CNN
    dqn.save(filepath=path_model)
