#### Flappy Bird policy selection function

#%%
# Imports
import numpy as np
import random

# We load the Q values dictionary (already learned)
Q_learned = np.load("Q.npy").item()

#%%
# Round function to define the grid
def myround(x):
    return int(5*round(float(x)/5))

# Key creation function to create a vector with the key variables
def getKey(pos, distance, vel):
    key = (myround(pos), myround(distance), vel)
    return key

#%%
# Function to select the optimal policy.
def FlappyPolicy(state, screen):
    
    # Current state's key
    pos = state["player_y"] - state["next_pipe_bottom_y"]
    distance = state["next_pipe_dist_to_player"]
    vel = state["player_vel"]        
    key = getKey(pos, distance, vel)
    
    if(Q_learned.get(key) == None):
        action = 119*random.randint(0,1) # In case key is non existent
    else:       
        action = 119*np.argmax(Q_learned[key])
        
    # We return the selected action
    return action