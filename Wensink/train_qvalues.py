## This Python script trains the FlappyBird using Q-Learning

from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np
import json
import random as rd

# Function to discretisize distances to 10x10 grids, which reduces the state space
def etat(status):
    
    xdif = status["next_pipe_dist_to_player"]
    ydif = status["next_pipe_bottom_y"] - status["player_y"]
    vel = status["player_vel"]
    
    if xdif < 140:
        xdif = int(xdif) - (int(xdif) % 10)
        
    else:
        xdif = int(xdif) - (int(xdif) % 70)

    if ydif < 180:
        ydif = int(ydif) - (int(ydif) % 10)
        
    else:
        ydif = int(ydif) - (int(ydif) % 60)

    estado =  str(int(xdif)) + '_' + str(int(ydif)) + '_' + str(int(vel))
 
    return estado         

#%%    

# Script to create Q-Value JSON file, initilazing with zeros
# DO NOT UNCOMMENT - YOU WOULD ERASE ALL EXISTING QVALUES INFORMATION
    
#qvalues = {}
## X -> [-40,-30...120] U [140, 210 ... 490]
## Y -> [-300, -290 ... 160] U [180, 240 ... 420]
#for x in chain(list(range(-40,140,10)), list(range(140,421,70))):
#    for y in chain(list(range(-300,180,10)), list(range(180,421,60))):
#        for v in range(-20,20):
#            qvalues[str(x)+'_'+str(y)+'_'+str(v)] = [0,0]
#            
#fd = open('qvalues.json', 'w')
#json.dump(qvalues, fd)
#fd.close()

# DO NOT UNCOMMENT - YOU WOULD ERASE ALL EXISTING QVALUES INFORMATION

#%%
    
#Constant values definition
reward = 0.0
nb_games = 10000
alpha =0.1
discount=0.99
bucle=0
epsilon = 0

#Loads qvalues information to a dictionary
qvalues = {}
fil = open('qvalues.json', 'r')
qvalues = json.load(fil)
fil.close()  

#Loads the FlappyBird game
game = FlappyBird(graphics="fixed") 
p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=True, display_screen=True)

p.init()

#Repeat the game nb_games times
for i in range(nb_games):
            
    p.reset_game()
   
    while(not p.game_over()):
        state = game.getGameState()
        bucle += 1
        current_state = etat(state)
         
        #As games are played, epsilon decreases 
        if bucle % 100 == 0:
            epsilon = epsilon * 0.9
        
        #Two options: epsilon < random implies deciding the following action from the qvalues
        if (epsilon < rd.random()):

            action_index = np.argmax(qvalues[current_state])
            action = action_index*119
        
        #Otherwise: action is random. Therefore, as epsilon decreases, qvalues are more often selected to decide the action
        else:
            action_index = rd.randint(0,1)
            action = action_index * 119
        
        #Get the reward        
        reward = p.act(action)
        
        #Get next state
        state_prima = game.getGameState()
        next_state = etat(state_prima)
        
        #Update qvalue with the qlearning algorithm formula
        qvalues[current_state][action_index] = (1 - alpha) * (qvalues[current_state][action_index]) + alpha * ( reward + discount*max(qvalues[next_state]))
        
#   Look for non  zero values in qvalues (useful the first times I trained the bird)      
#        for key, value in qvalues.items():
#            if value != [0,0]:
#                print(key,value) 
                
fd = open('qvalues.json', 'w')
json.dump(qvalues, fd)
fd.close()