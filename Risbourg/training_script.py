import numpy as np
from ple.games.flappybird import FlappyBird
from ple import PLE
import pickle

#=======================================================================================================================
# --  F U N C T I O N S  --


def get_action(a):
    actions = [0, 119]
    return actions[a]

# ----------------------------------------------------------------------------------------------------------------------

# Transforms the state into a unique string
# Keeps three parameters :
# x : Horizontal distance to the next pipe (chunks of 30 pixels when far enough, 10 otherwise)
# y = Vertical distance to the bottom of the next pipe (chunks of 30 pixels when far enough, 10 otherwise)
# v : player's vertical velocity
def map_state(state):
    # x
    if state['next_pipe_dist_to_player'] > 100:
        x = str(int(round(state['next_pipe_dist_to_player']/30)))
    else :
        x = '0' + str(int(round(state['next_pipe_dist_to_player']/10)))

    # y
    if abs(state['player_y'] - state['next_pipe_bottom_y']) > 60 :
        y = str(int(round((state['player_y'] - state['next_pipe_bottom_y'])/30)))
    else :
        y = '0' + str(int(round((state['player_y'] - state['next_pipe_bottom_y'])/10)))

    # v : player's vertical velocity
    v = str(int(state['player_vel']))

    return x + "_" + y + "_" + v

# ----------------------------------------------------------------------------------------------------------------------

# Returns the best action corresponding to the state (according to Q)
# Change of probability epsilon :
# Goes up when under the pipe with probability of 0.5 * epsilon
# Goes up (or opposite to Q's best option) with a probability of 0.1 otherwise
def epsilon_greedy(Q, s, epsilon, state):
    a = 0

    if s in Q.keys():
        a = np.argmax(Q[s][:])

    if np.random.rand() <= epsilon :
        if np.random.rand() <= 0.5 * epsilon:
            if state['next_pipe_bottom_y'] - state['player_y'] < 47 :
                a = 1
            else :
                a = 0
        else:
            if np.random.rand() <= 0.1:
                a = 1-a

    return a


#=======================================================================================================================
# --  M A I N  --

# Parameters
gamma = 0.95
alpha = 0.9
epsilon = 0.1

nb_games = 20000
resolution = 10

# Create Q
Q = dict()
# file = open("Qsarsa",'rb')
# Q = pickle.load(file)

# Create game
game = FlappyBird(graphics="fancy")
p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=True, display_screen=False)

# Count pipes
results_100 = 0
results_1000 = 0

# For each game
for g in range(1, nb_games):

    # print results of the last 50 games
    if g %100 == 0 :
        print('Moyenne des 100 derniers essais : %.2f' %(5 + results_100/100))
        if 5 + results_100/100 > 50:
            break
        results_100 = 0

    # Decrease alpha when learning
    if g %1000 == 0 :
        while alpha > 0.1 :
            alpha /= 1.01
        print('New epsilon is : %f at %d games played, with %d states explored' %(epsilon, g, len(Q)))
        print('Moyenne des 1000 derniers essais : %2f' % (5 + results_1000 / 1000))
        if results_1000 / 1000 > 40:
            break
        results_1000 = 0

    # Init new game
    p.init()
    p.reset_game()
    state = game.getGameState()
    # screen = p.getScreenRGB()
    reward = training_reward = 0

    # Init index
    s = map_state(state)
    action = epsilon_greedy(Q, s, epsilon, state)
    Q[s] = [0.0, 0.0]

    while not p.game_over():

        # play
        reward = p.act(get_action(action))

        # set training reward
        if reward == -5 :
            training_reward = -10000
        else:
            training_reward = 1

        # Get new states
        state_ = game.getGameState()
        # screen_ = game.getScreenRGB()
        s_ = map_state(state_)
        action_ = epsilon_greedy(Q, s_, epsilon, state_)

        # SARSA
        if s_ not in Q.keys():
            Q[s_] = [0.0, 0.0]
        delta = (training_reward + gamma * Q[s_][action_] - Q[s][action])
        Q[s][action] = Q[s][action] + alpha * delta

        # Update current states
        s = s_
        action = action_

        # Update result
        results_100 += reward
        results_1000 += reward

with open('Qsarsa', 'wb') as f:
    pickle.dump(Q,f)