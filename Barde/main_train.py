from deepqn import process_screen, epsilon, Agent, clip_reward, total_steps

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
if __name__ == "__main__":
    # create game
    game = FlappyBird(graphics="fixed")
    p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=True, display_screen=True)
    p.init()
    # initialises the scores and loss records
    average_score = 0
    average_scores = []
    max_scores = []
    loss_vec = []
    losses = []
    must_eval = False
    # defines the set of available actions
    ACTIONS = [119, None]
    # create the agent
    dqn = Agent()
    dqn.create_model()
    screen = process_screen(p.getScreenRGB())
    dqn.reset_state(screen)


    step = 0
    # Deep Q-learning with experience replay
    while step < total_steps:
        # p.act(ACTIONS[1]) # launches the game so we don't have a black screen

        # screen = process_screen(p.getScreenRGB())
        # dqn.reset_state(screen)

        if must_eval:
            average_score, max_score = dqn.evaluate_perfs(p, ACTIONS)
            average_scores.append(average_score)
            max_scores.append(max_scores)
            print("######################################")
            print("eval at step {} : mean = {}, max = {}".format(step, average_score, max_score))
            must_eval = False

        while not p.game_over():
            step += 1
            # dqn.display_state()
            # We select an action based on epsilon greedy policy
            a_idx = dqn.e_greedy_policy(step)
            # step
            r = p.act(ACTIONS[a_idx])
            r = clip_reward(r)
            dqn.store(screen, a_idx, r)

            # train
            if step > 10000:
                loss = dqn.learn(show=False)
                losses.append(loss)
                if step % 100 == 0:
                    loss_vec.append(np.mean(losses))
                    losses = []
                    print("mean loss at step {} : {} and epsilon : {}".format(step, loss_vec[-1], epsilon(step)))
                    # prepare next transition
                if step % 5000 == 0 and average_score < 20:
                    must_eval = True

            screen = process_screen(p.getScreenRGB())
            # print("Screen = {}".format(screen))
            dqn.update_state(screen)

        p.reset_game()
        # p.act(ACTIONS[1])
        # screen = process_screen(p.getScreenRGB())
        # dqn.update_state(screen)

    plt.subplot(2,2,(1,2))
    plt.plot(np.log(loss_vec))
    plt.subplot(223)
    plt.plot(average_scores)
    plt.subplot(224)
    plt.plot(max_scores)

    model_name = "new_train_{}".format(total_steps)
    print("saving the model : " + model_name)
    dqn.save_model(model_name)
    filehandler = open("new_train_losses_{}.pickle".format(total_steps), "wb")
    pickle.dump(loss_vec, filehandler)
    filehandler.close()
    # filehandler = open("full_mem_train_memory_{}.pickle".format(total_steps), "wb")
    # pickle.dump(replay_memory, filehandler)
    # filehandler.close()
    filehandler = open("new_train_scores_{}.pickle".format(total_steps), "wb")
    pickle.dump([average_scores, max_scores], filehandler)
    filehandler.close()
