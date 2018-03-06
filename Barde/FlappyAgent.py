from deepqn import DQN, process_screen
import numpy as np
import matplotlib.pyplot as plt

x_buffer = []
dqn = DQN()
dqn.load_model("Bis_no_clip_300000")
ACTIONS = [119, None]


def FlappyPolicy(state, screen):
    global x_buffer
    global dqn
    global ACTIONS

    screen = process_screen(screen)
    if len(x_buffer) == 0:
        x_buffer = [screen,screen,screen,screen]
    else :
        x_buffer.pop(0)
        x_buffer.append(screen)

    x = np.stack(x_buffer, axis=-1)

    # fig = plt.figure()
    # for j in range(4):
    #     ax = plt.subplot(1, 4, j + 1)
    #     ax.imshow(x[:, :, j], cmap="gray")
    #     ax.set_title("frame num {}".format(j))
    # fig.show()
    # input("Press Enter to continue...")

    a_idx = dqn.greedy_action(x)
    return ACTIONS[a_idx]