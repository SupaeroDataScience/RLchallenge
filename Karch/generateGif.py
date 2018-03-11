import numpy as np
from ple import PLE
from ple.games.flappybird import FlappyBird
from FlappyAgent import FlappyPolicy
import matplotlib

matplotlib.matplotlib_fname()
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from scipy import ndimage

import time

game = FlappyBird(graphics="fixed")
p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=True, display_screen=True)

p.init()

allscreens=[]
screens = []
count = 0
while (not p.game_over()):
    state = game.getGameState()
    screen = p.getScreenRGB()
    action = FlappyPolicy(state, screen)  ### Your job is to define this function.
    p.act(action)

    if count % 2 == 0:
        new_screen = np.zeros([512, 288, 3])
        for i in range(3):
            new_screen[:, :, i] = screen[:, :, i].T
            new_screen = np.array(new_screen, dtype=np.uint8)

        screens.append(new_screen)
    allscreens.append(screen)
    count += 1

    if count > 100:
        break

# fig = plt.figure()

# im = plt.imshow(new_screen, animated=True)
# plt.axis('off')


def updatefig(k):
    im.set_array(screens[k])
    return im,


# ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)
# ani.save('img/myanimation.gif', writer='imagemagick', fps=30)



