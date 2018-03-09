from deepqn import Agent, process_screen
import numpy as np
import matplotlib.pyplot as plt


dqn = Agent("lin")
dqn.load_model("my_trained_network")
ACTIONS = [119, None]


def FlappyPolicy(state, screen):
    global dqn
    global ACTIONS

    screen = process_screen(screen)
    if len(dqn.buffer_state) == 0:
        dqn.reset_state(screen)
    else:
        dqn.update_state(screen)
    a_idx = dqn.greedy_action()
    return ACTIONS[a_idx]