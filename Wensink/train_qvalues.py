# You're not allowed to change this file
from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np
import json
import random as rd
from itertools import chain



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


alpha =0.01
discount=0.99


#%%    
    # Script to create Q-Value JSON file, initilazing with zeros
    
qvalues = {}
# X -> [-40,-30...120] U [140, 210 ... 490]
# Y -> [-300, -290 ... 160] U [180, 240 ... 420]
for x in chain(list(range(-40,140,10)), list(range(140,421,70))):
    for y in chain(list(range(-300,180,10)), list(range(180,421,60))):
        for v in range(-20,20):
            qvalues[str(x)+'_'+str(y)+'_'+str(v)] = [0,0]
            
fd = open('qvalues.json', 'w')
json.dump(qvalues, fd)
fd.close()

#%%
reward = 0.0
nb_games = 100
cumulated = np.zeros((nb_games))
bucle=0

epsilon = 0.1

qvalues = {}
fil = open('qvalues.json', 'r')
qvalues = json.load(fil)
fil.close()  

game = FlappyBird(graphics="fixed") # use "fancy" for full background, random bird color and random pipe color, use "fixed" (default) for black background and constant bird and pipe colors.
p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=True, display_screen=True)
# Note: if you want to see you agent act in real time, set force_fps to False. But don't use this setting for learning, just for display purposes.

p.init()

for i in range(nb_games):
            
  
    #print(epsilon)
    p.reset_game()
    
    #print("GAME OVER")
   
    while(not p.game_over()):
        state = game.getGameState()
        screen = p.getScreenRGB()
        bucle += 1
        current_state = etat(state)
          
        if bucle % 50 == 0:
            epsilon = epsilon * 0.9


        if (epsilon < rd.random()):

            action_index = np.argmax(qvalues[current_state])
            action = action_index*119
            #if(qvalues[current_state][0] >= qvalues[current_state][1]):
               # action=119
                                    
            #else:
             #   action = 0
            
        else:
            action_index = rd.randint(0,1)
            action = action_index * 119
                
        reward = p.act(action)
        #print(p.game_over())
        #print("reward run", reward)
   
        state_prima = game.getGameState()
        
        next_state = etat(state_prima)
        
   
#        if next_state in qvalues:
#            pass
#        else:
#            #print("no existe")
#            qvalues[next_state] = [0,0]
            
        qvalues[current_state][action_index] = (1 - alpha) * (qvalues[current_state][action_index]) + alpha * ( reward + discount*max(qvalues[next_state]))
        
        #print(qvalues[current_state][action])
        
        #qvalues.update({(current_state) : [qvalues[current_state][0], qvalues[current_state][1]] })
#            if action_index == 0:
#                qvalues.update({(current_state) : [qvalues[current_state][action_index], int(qvalues[current_state][1])] })
#               
#            else:
#                qvalues.update({(current_state) : [int(qvalues[current_state][0]), qvalues[current_state][action]] })
            
#            #all(value == [0,0] for value in qvalues.values())
                
fd = open('qvalues.json', 'w')
json.dump(qvalues, fd)
fd.close()
#                        
#    for key, value in qvalues.items():
#        if value != [0,0]:
#            print(key,value)  
        #print('Q-values updated on local file.')
