
from deepqn import process_screen, MemoryBuffer, epsilon, DQN, clip_reward, total_steps,\
mini_batch_size, gamma, memory_size, epsilon_exp, epsilon_action

from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np
import pickle
import matplotlib.pyplot as plt

# xx = np.arange(total_steps)
# y = [epsilon(i) for i in xx]
# plt.plot(xx, y, color='r')
# # plt.plot(x, epsilon_action(x), color='b')
# plt.show()
# xx = None
# y = None

average_score = 0
scores = []

if __name__ == "__main__":
    game = FlappyBird(graphics="fixed")
    p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=True, display_screen=True)
    p.init()
    ACTIONS = [119, None]
    p.act(ACTIONS[1])
    # load or create model
    dqn = DQN()
    dqn.create_model()

    # initialize state and replay memory
    screen = process_screen(p.getScreenRGB())
    
    buffer_x = [screen, screen, screen, screen]
    x = np.stack(buffer_x, axis=-1)
    replay_memory = MemoryBuffer(memory_size)
    loss_vec = []
    losses = []
    # Deep Q-learning with experience replay
    for step in range(total_steps):

        # action selection
        if np.random.rand() < epsilon(step):
            if np.random.rand() < 0.5:
                a_idx = 0
            else:
                a_idx = 1
        else:
            a_idx = dqn.greedy_action(x)

        # step
        r = p.act(ACTIONS[a_idx])
        r = clip_reward(r)
        replay_memory.append(screen, a_idx, r)

        # train
        if step > 1001:
            X,A,R,Y,D = replay_memory.minibatch(mini_batch_size)
            QY = dqn.model.predict(Y)
            QYmax = QY.max(1)
            update = R + gamma * (1-D) * QYmax
            QX = dqn.model.predict(X)
            QX[np.arange(mini_batch_size), A.ravel()] = update.ravel()
            loss = dqn.model.train_on_batch(x=X, y=QX)
            losses.append(loss)
            if step % 1000 == 0:
                loss_vec.append(np.mean(losses))
                losses = []
                print("mean loss at step {} : {} and epsilon : {}".format(step, loss_vec[-1], epsilon(step)))
                # print("Q-value : \n {}".format(QY))
        # prepare next transition
            if step % 20000 == 0 and average_score < 20:
                nb_games = 10
                cumulated = np.zeros((nb_games))

                for ii in range(nb_games):
                    p.reset_game()
                    p.act(ACTIONS[np.random.randint(0, 2)])
                    screen = process_screen(p.getScreenRGB())
                    buffer_x = [screen, screen, screen, screen]
                    x = np.stack(buffer_x, axis=-1)

                    while not p.game_over():
                        screen = process_screen(p.getScreenRGB())
                        buffer_x.pop(0)
                        buffer_x.append(screen)
                        x = np.stack(buffer_x, axis=-1)
                        action = ACTIONS[dqn.greedy_action(x)]
                        reward = p.act(action)
                        cumulated[ii] = cumulated[ii] + reward

                average_score = np.mean(cumulated)
                max_score = np.max(cumulated)
                scores.append(average_score)

                print("Average score : {}".format(average_score))
                print("Max score : {}".format(max_score))
                # restart episode
                p.reset_game()
                p.act(ACTIONS[np.random.randint(0, 2)])
                screen = process_screen(p.getScreenRGB())
                buffer_x = [screen, screen, screen, screen]
                x = np.stack(buffer_x, axis=-1)

        if r < 0:
            # restart episode
            p.reset_game()
            p.act(ACTIONS[np.random.randint(0, 2)])
            screen = process_screen(p.getScreenRGB())
            # plt.imshow(screen, cmap="gray")
            # plt.show()
            buffer_x = [screen, screen, screen, screen]
            x = np.stack(buffer_x, axis=-1)
        else:
            # keep going
            screen = process_screen(p.getScreenRGB())
            buffer_x.pop(0)
            buffer_x.append(screen)
            x = np.stack(buffer_x, axis=-1)

    plt.subplot(211)
    plt.plot(np.log(loss_vec))
    plt.subplot(212)
    plt.plot(scores)

    model_name = "full_mem_train_{}".format(total_steps)
    print("saving the model : " + model_name)
    dqn.save_model(model_name)
    filehandler = open("full_mem_train_losses_{}.pickle".format(total_steps), "wb")
    pickle.dump(loss_vec, filehandler)
    filehandler.close()
    filehandler = open("full_mem_train_memory_{}.pickle".format(total_steps), "wb")
    pickle.dump(replay_memory, filehandler)
    filehandler.close()
    filehandler = open("full_mem_train_scores_{}.pickle".format(total_steps), "wb")
    pickle.dump(scores, filehandler)
    filehandler.close()
