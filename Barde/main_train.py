"""
Main file implementing the learning
"""
from deepqn import process_screen, epsilon, Agent, clip_reward, total_steps, epsilon_exp, epsilon_action, policy
from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np
import pickle
import matplotlib.pyplot as plt


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
    best_score = -5
    losses = []
    must_eval = False
    # defines the set of available actions
    ACTIONS = [119, None]
    # create the agent
    dqn = Agent(policy)
    dqn.load_model("retrain_10_to_5_clip_exp_reset_train_100000")
    filehandler = open("retrain_10_to_5_clip_exp_reset_train_memory_100000.pickle", "rb")
    dqn.memory = pickle.load(filehandler)
    filehandler.close()
    screen = process_screen(p.getScreenRGB())
    dqn.reset_state(screen)

    step = 0
    # Deep Q-learning with experience replay
    while step < total_steps:

        p.act(ACTIONS[np.random.randint(0, 2)])  # launches the game so we don't have a black screen
        screen = process_screen(p.getScreenRGB())
        dqn.update_state(screen)

        # we evaluate the performances of the agent
        if must_eval:
            average_score, max_score = dqn.evaluate_perfs(p, ACTIONS)
            average_scores.append(average_score)
            max_scores.append(max_score)
            if average_score > best_score :
                dqn.save_model("bis_best_for_now_{}_with_{}".format(step, average_score))
                best_score = average_score
            print("######################################\n")
            print("######################################\n")
            print("Eval at step {} : mean = {}, max = {}".format(step, average_score, max_score))
            print("######################################\n")
            print("######################################\n")
            must_eval = False

        # we do one game
        while not p.game_over():
            step += 1
            # dqn.display_state()

            # We select an action based on epsilon greedy policy
            a_idx = dqn.policy(step)

            # we act one step
            r = p.act(ACTIONS[a_idx])
            r = clip_reward(r)
            dqn.store(screen, a_idx, r)

            # training process
            if step > 0:
                loss = dqn.learn(show=False)
                losses.append(loss)
                if step % 500 == 0:
                    loss_vec.append(np.mean(losses))
                    losses = []
                    if policy == "exp_action":
                        print("############################ Iteration {} ############################".format(step))
                        print(" Mean loss : {} \n Epsilon : {} \n Epsilon Action : {}".format(loss_vec[-1],
                                                                                              epsilon_exp(step),
                                                                                              epsilon_action(step)))
                        print("\n")
                    elif policy == "lin":
                        print("############################ Iteration {} ############################".format(step))
                        print(" Mean loss : {} \n Epsilon : {}".format(loss_vec[-1], epsilon(step)))
                        print("\n")
                        # prepare next transition
                if step % 25000 == 0:
                    must_eval = True

            screen = process_screen(p.getScreenRGB())
            # print("Screen = {}".format(screen))
            dqn.update_state(screen)

        p.reset_game()

    model_name = "bis_retrain_const_5_clip_exp_reset_train_{}".format(total_steps)
    print("saving the weights : " + model_name)
    dqn.save_weights("weights_" + model_name)
    print("saving the model : " + model_name)
    dqn.save_model(model_name)
    filehandler = open("bis_retrain_const_5_clip_exp_reset_train_losses_{}.pickle".format(total_steps), "wb")
    pickle.dump(loss_vec, filehandler)
    filehandler.close()
    filehandler = open("bis_retrain_const_5_clip_exp_reset_train_memory_{}.pickle".format(total_steps), "wb")
    pickle.dump(dqn.memory, filehandler)
    filehandler.close()
    filehandler = open("bis_retrain_const_5_clip_exp_reset_train_scores_{}.pickle".format(total_steps), "wb")
    pickle.dump([average_scores, max_scores], filehandler)
    filehandler.close()

    plt.subplot(2, 2, (1, 2))
    plt.plot(np.log(loss_vec))
    plt.subplot(223)
    plt.plot(average_scores)
    plt.subplot(224)
    plt.plot(max_scores)
    plt.show()