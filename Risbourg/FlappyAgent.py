import pickle
import numpy as np

has_q = 0
Q = dict()

#=======================================================================================================================
# --  F U N C T I O N S  --

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

#=======================================================================================================================
# --  P O L I C Y  --

def FlappyPolicy(state, screen):

    global has_q
    global Q

    if has_q == 0 :
        file = open("Qsarsa",'rb')
        Q = pickle.load(file)
        has_q = 1

    s = map_state(state)

    if s in Q.keys():
        a = np.argmax(Q[s][:])
    else:
        a = 0

    return 119 if a else 0
