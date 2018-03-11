
import numpy as np
Q=np.load("trained_Q.npy")

def FlappyPolicy(state, screen):

    # Using "state"
    y = int(288 + (state['next_pipe_top_y'] + state['next_pipe_bottom_y']) * 0.5 - state['player_y'])
    x = int(state['next_pipe_dist_to_player'])
    v = int(state['player_vel'])
                
    action=None
    action = int(np.argmax(Q[y][x][v][:]))
    if (action == 1): 
            action = 119
          
    return action
