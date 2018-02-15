# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 14:10:25 2018

@author: Julio Hernanz González
"""

#### Creation of the Q-value dictionary for the Flappy Bird PLE game.
#### This is the training file. The bird plays the game a NB_GAMES and 
#### saves the information, learning using the Q-learning algorithm.

#%% Imports

from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np
import random

#%% Functions

# Function to create a grid for x and y position values.
def myround(x):
    return int(5 * round(float(x)/5))

# Creation of a vector with the three variables that are saved as key of the dictionary.
def getKey(pos, distance, vel):
    key = (myround(pos), myround(distance), vel)
    return key

#%% Q-learning code

# Beginning the game :
game = FlappyBird(graphics="fixed")
p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=False, display_screen=True)
p.init() # Initialize the game
reward = 0.0

NB_GAMES = 1000 # Number of games to be played
cumulated = np.zeros((NB_GAMES))
counter = 0
games_counter = 0 

# We initialize or load the Q dictionary
#Q = dict() # to restart the dictionary
Q = np.load("Q.npy").item() # to keep learning on an existing dictionary

# Q learning parameters
GAMMA = 0.9
ALPHA = 0.01
EPSILON = 0.01
ACTIONS = [0, 119] # allowed actions for the game (nothing / flap)

for i in range(NB_GAMES):
    p.reset_game()    
    games_counter += 1
                
    if(games_counter % 500 == 0):
        print(games_counter) # to keep track of how many games have been played
    
    while(not p.game_over()):
        state = game.getGameState() # Current state.
        
        # Key of the current state
        pos = state["player_y"] - state["next_pipe_bottom_y"] # y position difference between the bid and bottom pipe.
        distance = state["next_pipe_dist_to_player"] # x distance to the pipes
        vel = state["player_vel"] # bird speed
        key = getKey(pos, distance, vel)
        
        # In case key is non existent we create an entry in the dictionary
        if(Q.get(key) == None):
            Q[key] = [0 , 0]
        
        counter += 1
        
        if(counter % 500 == 0): # to reduce the value of epsilon 
            EPSILON *= 0.9

        # This if allows to select actions randomly at the beginning and go 
        # back to already learned states afterwards.
        if(EPSILON < random.random()):
            a = np.argmax(Q[key])
            action = ACTIONS[a]
        else:
            a = random.randint(0,1)
            action = ACTIONS[a]
 
        # We execute the action and save the reward value.
        reward = p.act(action)
        cumulated[i] = cumulated[i] + reward

        # Next state and its key
        state = game.getGameState()
        posprima = state["player_y"] - state["next_pipe_bottom_y"]
        distanceprima = state["next_pipe_dist_to_player"]
        velprima = state["player_vel"]
        keyprima = getKey(posprima, distanceprima, velprima)
        
        # In case key is non existent we create an entry on the dictionary
        if(Q.get(keyprima) == None):
            Q[keyprima] = [0 , 0]
            
        # We select the best next possible action's Q value
        maxQsprima = max(Q[keyprima])
        
        # We update the Q value
        Q[key][a] = (1-ALPHA)*Q[key][a] + ALPHA*(reward + GAMMA*maxQsprima)
        
    # Next state when p.game_over() and its key
    state = game.getGameState()
    posprima = state["player_y"] - state["next_pipe_bottom_y"]
    distanceprima = state["next_pipe_dist_to_player"]
    velprima = state["player_vel"]
    keyprima = getKey(posprima, distanceprima, velprima)
    
    if(Q.get(keyprima) == None):
        Q[keyprima] = [0 , 0]

    maxQsprima = max(Q[keyprima])
    Q[key][a] = Q[key][a] + ALPHA*(reward + GAMMA*maxQsprima - Q[key][a])

max_score = np.max(cumulated)
print(max_score)

# We save the Q function
np.save("Q.npy",Q)