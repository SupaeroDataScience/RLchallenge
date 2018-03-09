

import random 
import numpy as np

#WE IMPORT AN ALREADY CREATED FILE WITH THE Q MATRIX
MatrizQ_readed = np.load("QM.npy").item()


#WE DEFINE THE FlappyPolicy FUNCTION
def FlappyPolicy(state, screen):
    
    
        actions = [0,119]
        
        #WE USE ONLY FOUR DATA OF THE BIRD'S STATE
        disty=state["next_pipe_bottom_y"]-state["player_y"]
        distx= state["next_pipe_dist_to_player"]
        velp=state["player_vel"]
        
        #WE DISCRTIZE THE  BIRD'S STATE
        if distx < 3000:
            distx = int(distx) - (int(distx) % 20)
        else:
            distx = 3000 

        if disty < 1800:
            disty = int(disty) - (int(disty) % 20)
        else:
            disty = 1800
        
        
        #WE DISCRITIZE THE BIRD'S SPEED
        if velp < 1800:
            velp = int(velp) - (int(velp) % 2)
        else:
            velp = 1800
        
        #WE CREATE A DICTIONARY KEY FOR THE PRECEDENT BIRD'S STATE AND SPEED
        stringdisty= str(disty)
        stringdistx= str(distx)
        stringvelp= str(velp)
        statep= "".join ([stringdisty," / ", stringdistx," / ", stringvelp])
        #print(statep)


        #WE ESTABLISH THE BIRD'S ACTION
        if(MatrizQ_readed.get(statep) == None):
            bestindex = random.randint(0,1)
            action = bestindex * 119
            
        else:       
            bestindex = np.argmax(MatrizQ_readed[statep])
            action = actions[bestindex]
            
        
        #THE PRECEDENT ACTION IS RETURNED
        return action
