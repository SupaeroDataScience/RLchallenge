# You're not allowed to change this file
from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np
from FlappyAgent import FlappyPolicy

game = FlappyBird()
p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=True, display_screen=True)
# Note: if you want to see you agent act in real time, set force_fps to False. But don't use this setting for learning, just for display purposes.

p.init()
reward = 0.0

nb_games = 5
cumulated = np.zeros((nb_games))

for i in range(nb_games):
    p.reset_game()
    
    while(not p.game_over()):
        state = game.getGameState()
        screen = p.getScreenRGB()
        action=FlappyPolicy(state, screen) ### Your job is to define this function.
        #print(action)
        reward = p.act(action)
        cumulated[i] = cumulated[i] + reward

average_score = np.mean(cumulated)
max_score = np.max(cumulated)

print('avg:',average_score)
print('max:',max_score)