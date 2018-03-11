import json
import numpy as np

fil = open('qvalues.json', 'r')
qvalues = json.load(fil)
fil.close() 

def FlappyPolicy(state,screen):
    
    current_state = etat(state)
    action_index = np.argmax(qvalues[current_state])
    action = action_index*119
        
    return action
        
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
    
   
        
            


