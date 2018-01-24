import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import resize

from DumbPolicy import FlappyPolicy1
from MiddlePolicy import FlappyPolicy2
from SmartPolicy import Policy

def FlappyPolicy(state, screen):
    difficulty = 3
    if difficulty == 1:
        return FlappyPolicy1(state, screen)

    elif difficulty == 2:
        return FlappyPolicy2(state, screen)

    elif difficulty == 3:
        p = Policy(state, screen)

        # Cut, grey, resize and stack (84, 84, 4)
        # s, new screen to take into account
        s = p.transform_screen()

        # load previous learning

        # Train and play

        # Save potential learning

        # Choose action
        action = p.get_action()
        return action
    else :
        return None










