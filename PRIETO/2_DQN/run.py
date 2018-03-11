# You're not allowed to change this file
from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np
from FlappyAgent import FlappyPolicy, process_screen
from collections import deque
from keras.models import load_model

game = FlappyBird(graphics="fixed") # use "fancy" for full background, random bird color and random pipe color, use "fixed" (default) for black background and constant bird and pipe colors.
p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=True, display_screen=True)
# Note: if you want to see you agent act in real time, set force_fps to False. But don't use this setting for learning, just for display purposes.

p.init()
reward = 0.0

nb_games = 100
cumulated = np.zeros((nb_games))

Q = load_model("DQN.h5")

print(Q.summary())

for i in range(nb_games):
    p.reset_game()
    X = deque([np.zeros((80, 80)),
                       np.zeros((80, 80)),
                       np.zeros((80, 80)),
                       np.zeros((80, 80))], maxlen=4)

    while(not p.game_over()):
        state = game.getGameState()

        # Process screen
        s = process_screen(p.getScreenRGB())

        X.append(s)

        action = FlappyPolicy(state,X,Q) ### Your job is to define this function.
        
        reward = p.act(action)
        cumulated[i] = cumulated[i] + reward
        start = False

average_score = np.mean(cumulated)
max_score = np.max(cumulated)

print(f"average_score: {average_score}")
print(f"max_score: {max_score}")