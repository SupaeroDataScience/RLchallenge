import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import resize


def FlappyPolicy(state, screen):
    difficulty = 1
    if difficulty == 1:
        return FlappyPolicy1(state, screen)
    elif difficulty == 2:
        return FlappyPolicy2(state, screen)
    elif difficulty == 3:
        return FlappyPolicy3(state, screen)
    else :
        return None


def FlappyPolicy1(state, screen):
    action = None
    y = state['player_y']
    if y >= state['next_pipe_bottom_y'] - 50 :
        action = 119

    return action


def FlappyPolicy2(state, screen):
    action = None

    return action


def FlappyPolicy3(state, screen):
    # by default : no action
    action = None

    # analyze of the screen
    # Transform in grey + small size pixels
    y = transform_screen(screen)
    plt.imshow(y, cmap="gray")
    plt.show()

    if (np.random.randint(0, 7) < 1):
        action = 119 #up

    return action


def transform_screen(screen):
    print(screen.shape)
    output = 256*(rgb2gray(screen))[:, 0:400]
    output = resize(output, (84, 84))
    print(output.shape)
    return output

