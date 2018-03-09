import numpy as np

# grid definition
resolution = 20
next_pipe_bottom_y = np.array(range(0,400,resolution))
player_vel = np.array(range(-10,10, 2))
player_y = np.array(range(0,400,resolution))
next_pipe_dist_to_player = np.array(range(0,300,resolution))

with open('Qsarsa', 'wb') as f:
    pickle.dump(Q,f)

def QToPolicy(Q):
    pi = np.zeros((len(player_y), len(next_pipe_bottom_y), len(player_vel), len(next_pipe_dist_to_player)))
    for i in range(len(player_y)):
        for j in range(len(next_pipe_bottom_y)):
            for k in range(len(player_vel)):
                for l in range(len(next_pipe_dist_to_player)):
                    pi[i][j][k][l] = np.argmax(Q[i][j][k][l][:])
    return pi

def FlappyPolicy(state, screen):
    
    # determine state
    i = np.argmin(abs(player_y - state['player_y']))
    j = np.argmin(abs(next_pipe_bottom_y - state['next_pipe_bottom_y']))
    k = np.argmin(abs(player_vel - state['player_vel']))
    l = np.argmin(abs(next_pipe_dist_to_player - state['next_pipe_dist_to_player']))
    
    action = ToAction(Pisarsa[i][j][k][l])
    return action

file = open("Qsarsa",'rb')
Qsarsa = pickle.load(file)
Pisarsa = QToPolicy(Qsarsa)



