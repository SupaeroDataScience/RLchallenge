def new_state(state):

    x = int(round((state['next_pipe_dist_to_player'])/20))
    y = int(round((state['player_y'] - state['next_pipe_bottom_y'])/20))+15
    v = int(state['player_vel'])+10

    return [x,y,v]
