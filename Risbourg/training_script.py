import numpy as np
from ple.games.flappybird import FlappyBird
from ple import PLE
import pickle

#=======================================================================================================================
# --  F U N C T I O N S  --

# Gets the action order to send to the game :
# 0 -> 0 : don't flap
# 1 -> 119 : flap
def get_action(a):
    return 119*a

# ----------------------------------------------------------------------------------------------------------------------

# Transforms the state into a string
# Keeps three parameters :
# x : Horizontal distance to the next pipe (chunks of 30 pixels when far enough, 10 otherwise)
# y = Vertical distance to the bottom of the next pipe (chunks of 30 pixels when far enough, 10 otherwise)
# v : player's vertical velocity
def map_state(state):
    if state['next_pipe_dist_to_player'] > 100:
        x = str(int(round(state['next_pipe_dist_to_player']/30)))
    else :
        x = '0' + str(int(round(state['next_pipe_dist_to_player']/10)))

    if abs(state['player_y'] - state['next_pipe_bottom_y']) > 60 :
        y = str(int(round((state['player_y'] - state['next_pipe_bottom_y'])/30)))
    else :
        y = '0' + str(int(round((state['player_y'] - state['next_pipe_bottom_y'])/10)))

    v = str(int(state['player_vel']))

    return x + "_" + y + "_" + v

# ----------------------------------------------------------------------------------------------------------------------

# Returns the best action corresponding to the state (according to Q)
# With probability epsilon :
# Flaps when under the pipe with probability of 0.5 * epsilon
# Flaps (or opposite to Q's best option) with a probability of 0.1 otherwise
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
# Gamma (discount-factor) is very high because the game is deterministic
gamma = 0.95
# Alpha (learning rate) is initialized very high to increase the learning rate at the beginning of the training
alpha = 0.9
# Epsilon is used for the epsilon-greedy function. Set not to high so as to continue exploring while improving the
# explored states
epsilon = 0.1

# 20 000 games is enough but the training will stop before that if a score good enough is found
nb_games = 20000

# Create Q
Q = dict()

# Create game
game = FlappyBird(graphics="fancy")
p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=True, display_screen=False)

# Count the results to print them every 100 and 1000 games
results_100 = 0
results_1000 = 0

# For each game
for g in range(1, nb_games):


    if g %100 == 0 :
        # print results of the last 100 games
        print('Moyenne des 100 derniers essais : %.2f' %(5 + results_100/100))
        # Stops the training when we reach an average score superior to 50 on the last 100 games
        if 5 + results_100/100 > 50:
            break
        results_100 = 0

    if g %1000 == 0 :
        # Decrease alpha every 1000 games when learning
        while alpha > 0.1 :
            alpha /= 1.01
        # print some information and results of the last 1000 games
        print('%d games played, with %d states explored' %(g, len(Q)))
        print('Moyenne des 1000 derniers essais : %2f' % (5 + results_1000 / 1000))
        results_1000 = 0

    # Initialize new game
    p.init()
    p.reset_game()
    state = game.getGameState()
    # We don't get the screen in order to decrease the processing time of each iteration
    # screen = p.getScreenRGB()
    reward = training_reward = 0

    # Initialize state and action
    s = map_state(state)
    action = epsilon_greedy(Q, s, epsilon, state)
    Q[s] = [0.0, 0.0]

    while not p.game_over():

        # play
        reward = p.act(get_action(action))

        # set training reward
        # Training reward is set to -10 000 when the bird dies (the action should never be used in this state)
        if reward == -5 :
            training_reward = -10000
        # Training reward is set to 1 as long as the bird lives. We don't take into account the number of pipes passed
        # because it has no impact on the quality of an action in a given state
        else:
            training_reward = 1

        # Get new states
        state_ = game.getGameState()
        # screen_ = game.getScreenRGB()
        s_ = map_state(state_)
        action_ = epsilon_greedy(Q, s_, epsilon, state_)

        # SARSA algorithm with TD(0)
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

# Saves Q in a file
with open('Qsarsa', 'wb') as f:
    pickle.dump(Q,f)