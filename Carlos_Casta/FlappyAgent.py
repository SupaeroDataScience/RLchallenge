

import random 
import numpy as np


MatrizQ_readed = np.load("QMATRIX.npy").item()



def FlappyPolicy(state, screen):
    
    
        actions = [0,119]
        disty=state["next_pipe_bottom_y"]-state["player_y"]
        distx= state["next_pipe_dist_to_player"]
        velp=state["player_vel"]
        
        if distx < 3000:
            distx = int(distx) - (int(distx) % 10)
        else:
            distx = 3000 

        if disty < 1800:
            disty = int(disty) - (int(disty) % 20)
        else:
            disty = 1800
        
        #if velp < 1800:
           # velp = int(velp) - (int(velp) % 2)
       # else:
           # velp = 1800
        
        
        stringdisty= str(disty)
        stringdistx= str(distx)
        stringvelp= str(velp)
        
        statep= "".join ([stringdisty," / ", stringdistx," / ", stringvelp])
        #print(statep)

        if(MatrizQ_readed.get(statep) == None):
            bestindex = random.randint(0,1)
            action = bestindex * 119
            
        else:       
            bestindex = np.argmax(MatrizQ_readed[statep])
            action = actions[bestindex]
            
        
 
        return action
