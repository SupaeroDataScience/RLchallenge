import numpy as np
#state: type dict
#   player_y 
#   player_vel
#   next_pipe_dist_to_player
#   next_pipe_top_y
#   next_pipe_bottom_y
#   next_next_pipe_dist_to_player
#   next_next_pipe_top_y
#   next_next_pipe_bottom_y

#screen: RGB 
#   screen: np.array(288,512,3) 

#action: None or 119

#Initial state: writes as state['string']
    #{'player_y': 256, 'player_vel': 0, 'next_pipe_dist_to_player': 283, 'next_pipe_top_y': 53, 'next_pipe_bottom_y': 153, 'next_next_pipe_dist_to_player': 427.0, 'next_next_pipe_top_y': 153, 'next_next_pipe_bottom_y': 253}

def FlappyPolicy(state, screen,p):
    action=None
    print(p.act(action))
    if(np.random.randint(0,2)<1):
        action=119
    return action


