import os
import numpy as np
from ple import PLE
from ple.games.flappybird import FlappyBird
from algorithms import (FeaturesNeuralQLearning, FeaturesLambdaSarsa,
                        DeepQLearning, DISPLAY)

ACTIONS = [None, 119]
if not DISPLAY:
    os.environ['SDL_VIDEODRIVER'] = 'dummy'


def test_play(agent, game, n, accelerated=False):
    p = PLE(game, fps=30, frame_skip=1, num_steps=1,
            force_fps=accelerated, display_screen=DISPLAY)
    cumulated = np.zeros(n, dtype=np.int32)
    for i in range(n):
        p.reset_game()
        while not p.game_over():
            state = game.getGameState()
            qvals = agent.get_qvals(state)
            act = agent.greedy_action(qvals, 0)
            reward = p.act(ACTIONS[act])
            if reward > 0:
                cumulated[i] += 1
        print('Game:', i, ', doors:', cumulated[i])
    average_score = np.mean(cumulated)
    max_score = np.max(cumulated)
    min_score = np.min(cumulated)
    print('\nTest over', n, 'tests:')
    print('average_score', 'max_score', 'min_score\n',
          average_score, max_score, min_score)
    return average_score, max_score, min_score


if __name__ == '__main__':
    game = FlappyBird()

    agent = FeaturesLambdaSarsa()
    # agent = FeaturesNeuralQLearning()
    # agent = DeepQLearning()

    # agent.train(True, game, DISPLAY)
    agent.load()

    average_score, max_score, min_score = test_play(agent, game, 10, True)
    # test_play(agent, game, 2, False)
