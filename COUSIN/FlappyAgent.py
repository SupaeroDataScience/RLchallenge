from collections import deque
from keras.models import load_model
import numpy as np
import PARAMS as params
from Tools import process_screen, greedy_action

# keep some information about previous games + load only once the model
shape_img = params.SIZE_IMG
dqn = load_model("Save/model_dqn_flappy3.h5")
frames = deque([np.zeros(shape_img), np.zeros(shape_img), np.zeros(shape_img), np.zeros(shape_img)], maxlen=4)


def FlappyPolicy(state, screen):
    """ Policy based on a Deep Q Network. Use pixels.

    :param state (not used)
    :param screen
    :return action (no jump (0) or jump (119))
    """
    global dqn
    global frames
    global shape_img

    screen_x = process_screen(screen)
    print(screen_x)
    print()

    # a game starts again + black screen
    if np.count_nonzero(screen_x) < 5:  # we take a 5 pixels error / Should be equal to zero
        frames = deque([np.zeros(shape_img), np.zeros(shape_img), np.zeros(shape_img), np.zeros(shape_img)], maxlen=4)

    frames.append(screen_x)
    x = np.stack(frames, axis=-1)
    a = greedy_action(dqn, x)  # 0 or 1

    return params.LIST_ACTIONS[a]
