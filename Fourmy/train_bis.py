import os
import numpy as np
from ple import PLE
from ple.games.flappybird import FlappyBird
from train import FeaturesNeuralQLearning, FeaturesLambdaSarsa

# TODO: params.py?
ACTIONS = [None, 119]
DISPLAY = True
if not DISPLAY:
    os.environ['SDL_VIDEODRIVER'] = 'dummy'



def test_play(agent, n, accelerated=False):
    p = PLE(agent.game, fps=30, frame_skip=1, num_steps=1,
            force_fps=accelerated, display_screen=DISPLAY)
    cumulated = np.zeros(n, dtype=np.int32)
    for i in range(n):
        p.reset_game()
        while not p.game_over():
            state = agent.game.getGameState()
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

    # agent = FeaturesNeuralQLearning(game, DISPLAY)
    agent = FeaturesLambdaSarsa(game, DISPLAY)

    # agent.train(scratch=True)
    agent.load()

    average_score, max_score, min_score = test_play(agent, 10, True)
    test_play(agent, 2, False)
