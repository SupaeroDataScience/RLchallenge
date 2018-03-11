

from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np
import matplotlib.pyplot as plt

game = FlappyBird()
p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=True, display_screen=False)

p.init()
reward = 0.0
nb_games = 10000
cumulated = np.zeros((nb_games))

# parameter of modele
r_1 = 1
r_2 = -100
alpha = 0.04 

x_wall = np.zeros((40))
y_wall = np.zeros((40))
v_wall = np.zeros((40))
a_wall = np.zeros((40))
#Q(y,x,v,a) ,a is set of action
Q = np.zeros((512,300, 21, 2))
## fly if y < 273
Q[255:511,:,:,0] = 0.1
Q[0:254,:,:,1] = 0.1
# between the pipe
Q[:,8,:,1] = 0.2  # in the middle: jump
Q[216:256,120:144,:,1] = 0.2  # jump if too low
Q[256:306,120:144,:,0] = 0.2

for i in range(nb_games):
    p.reset_game()
            
    while(not p.game_over()):
        state = game.getGameState()
        screen = p.getScreenRGB()
        #instead of using absolute position of pipe, use relative position
        y = int(288 + (state['next_pipe_top_y'] + state['next_pipe_bottom_y']) * 0.5 - state['player_y'])
        x = int(state['next_pipe_dist_to_player'])
        v = int(state['player_vel'])
        
        #greedy policy
        action = int(np.argmax(Q[y][x][v][:]))
        if (action == 1): 
            action_value = 119 
        else: action_value=None        
        if (i>1):
            for j in range(37-1, 0, -1):
                x_wall[j] = int(x_wall[j-1])
                y_wall[j] = int(y_wall[j-1])
                v_wall[j] = int(v_wall[j-1])
                a_wall[j] = int(a_wall[j-1])
            x_wall[0] = int(x)
            y_wall[0] = int(y)
            v_wall[0] = int(v)
            a_wall[0] = int(action)
       
        #reward is +1 if bird fly by the pipe
        reward = p.act(action_value)
        my_reward=0
        if (reward==1):
            my_reward = r_1
            cumulated[i] += 1
            for j in range(1, 40):
                Q[int(y_wall[j]),int(x_wall[j]),int(v_wall[j]),int(a_wall[j])] += alpha * (my_reward + np.max(Q[int(y_wall[j-1]),int(x_wall[j-1]),int(v_wall[j-1]),int(a_wall[j-1])]))
        
        # bad result : -100
        if (reward<0):
            my_reward = r_2
            if (x==20):
                for j in range(0, 27):
                    Q[int(y_wall[j]),int(x_wall[j]),int(v_wall[j]),int(a_wall[j])] += alpha * (my_reward + np.max(Q[int(y_wall[j-1]),int(x_wall[j-1]),int(v_wall[j-1]),int(a_wall[j-1])]))
            else:
               for j in range(0, 6):
                    Q[int(y_wall[j]),int(x_wall[j]),int(v_wall[j]),int(a_wall[j])] += alpha * (my_reward + np.max(Q[int(y_wall[j-1]),int(x_wall[j-1]),int(v_wall[j-1]),int(a_wall[j-1])]))

np.save('trained_Q', Q)
    

