import numpy as np
from ple import PLE
# You're not allowed to change this file
from ple.games.flappybird import FlappyBird

from Cattelle.FlappyAgent import FlappyPolicy

game = FlappyBird(
    graphics="fixed")  # use "fancy" for full background, random bird color and random pipe color, use "fixed" (
# default) for black background and constant bird and pipe colors.
p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=True, display_screen=True)
# Note: if you want to see you agent act in real time, set force_fps to False. But don't use this setting for
# learning, just for display purposes.

p.init()
reward = 0.0

nb_games = 100
cumulated = np.zeros(nb_games)

try:
    for i in range(nb_games):
        p.reset_game()

        while not p.game_over():
            state = game.getGameState()
            screen = p.getScreenRGB()
            action = FlappyPolicy(state, screen)  # Your job is to define this function.

            reward = p.act(action)
            cumulated[i] = cumulated[i] + reward

except Exception as e:
    # No matter what the program stopped, we still want our top and average score for debugging purposes
    # print(f'Received exception "{e}"')
    raise e

average_score = np.mean(cumulated)
max_score = np.max(cumulated)

print("Average score:", average_score)
print("Max score:", max_score)
