import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import resize

from DumbPolicy import FlappyPolicy1
from MiddlePolicy import FlappyPolicy2
from SmartPolicy import Policy3

DIFFICULTY = 3

def FlappyPolicy(state, screen):
    ''' Function to play the final exam ! '''

    if DIFFICULTY == 1:
        return FlappyPolicy1(state, screen)

    elif DIFFICULTY == 2:
        return FlappyPolicy2(state, screen)

    elif DIFFICULTY == 3:
        path_model = "Save/model_dql_flappy3_dense.h5"
        possible_actions = [0, 119]
        p = Policy3(state, screen, path_model)

        # Cut, grey, resize and stack (84, 84, 4)
        # s, new screen to take into account
        transformed_screen = p.transform_screen() #screen to use in CNN for choice

        # Choose action
        action_ind = p.get_action(transformed_screen)
        print(action_ind)
        return possible_actions[action_ind]
    else:
        return None










