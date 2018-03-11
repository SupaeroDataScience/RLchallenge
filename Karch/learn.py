import numpy as np
from ple import PLE
from ple.games.flappybird import FlappyBird

from FlappyAgent import FlappyAgent

game = FlappyBird(graphics="fixed")
p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=True, display_screen=True)
p.init()
reward = 0.0

nb_games = 100000
cumulated = np.zeros((nb_games))

agent = FlappyAgent(84, 2)

try :
    agent.importMemory('human_policy/human_replay.pkl')
except IOError:
    print("!!!!!!!!!!!!!!!!!!! WARNING !!!!!!!!!!!!!!!!!!!!")
    print("Learning without human experience initialization")
    print("!!!!!!!!!!!!!!!!!!! WARNING !!!!!!!!!!!!!!!!!!!!")

# Defining the variables we want to monitor
avg_costlist = [0]
max_costlist = [0]
scoreslist = []
cost_tmp = []
test_game = False

highest_score = 20

# Signature to write results
comment = 'test'

count = 0
for i in range(nb_games):
    p.reset_game()

    if test_game:
        scores = agent.evaluateScore(20, p)
        scoreslist.append(scores)
        print("--------------------------------")
        print("------------- TEST -------------")
        print("Average_score : {}".format(np.mean(scores)))
        if np.mean(scores) > highest_score :
            agent.save('results/network'+comment+str(count))
            highest_score=np.mean(scores)
        test_game = False

    while (not p.game_over()):
        # state = game.getGameState()
        screen = p.getScreenRGB()
        action = agent.act(screen)
        reward = p.act(action)
        count += 1
        agent.remember(screen, action, reward)

        if len(agent.memory) > agent.batch_size:
            cost = agent.replay()
            cost_tmp.append(cost)

        cumulated[i] = cumulated[i] + reward

        if count % 200 == 0:
            avg_costlist.append(np.mean(cost_tmp))
            max_costlist.append(np.max(cost_tmp))
            cost_tmp = []

        if count % 25000 == 0:
            print("testing game")
            test_game = True

    print("--------------------------------")
    print("Game : {} - Score : {}".format(i, cumulated[i]))
    print("ε : {} ".format(agent.epsilon))
    print("Iteration : {}".format(count))
    print("cost : {}".format(avg_costlist[-1]))

    if count > 300000:
        break

# saving results

np.save('results/avg_cost' + comment, np.array(avg_costlist))
np.save('results/max_cost' + comment, np.array(max_costlist))
np.save('results/scores' + comment, np.array(scoreslist))
agent.save('results/network' + comment)
average_score = np.mean(cumulated)
max_score = np.max(cumulated)
