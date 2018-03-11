# -*- coding: utf-8 -*-
"""
Imported when the main run is launched

Load a trained tree from  the 'trained_tree.txt' file 

FlappyPolicy :
    Return the best action to perform based on UCB1 method, 
    given the current state of the bird

No training is performed anymore at this state

@author: Guillaume Plessix
"""
import FlappyMcts

#Location of the trained tree file to load 
trained_tree_address = 'trained_tree'


main_tree = {}
main_tree = FlappyMcts.load_tree(main_tree,trained_tree_address)

def etat_discret_state(state):
    
    scale_dist_to_player = 4
    scale_hauteur = 8

    player_vel = state['player_vel']
    next_pipe_dist_to_player = int(state['next_pipe_dist_to_player']/scale_dist_to_player)
    hauteur = int((int(state['player_y']) -int(state['next_pipe_top_y']))/scale_hauteur)   
    
    return (player_vel, next_pipe_dist_to_player, hauteur)




def FlappyPolicy(state, screen):
    
    current_state = etat_discret_state(state)
    
    #Get the graph node matching the current state
    current_node = FlappyMcts.search_node(main_tree, current_state)

    if current_node == None : #The node does not exists in the tree : it is the first time the state is visited
        current_node = FlappyMcts.node(current_state) #A new node is created
        main_tree[current_state] =  current_node #The node is added to the tree

    #Action is chosen based on UCB1 method
    action = FlappyMcts.selection_expansion(main_tree, current_state)
    return(action)
    
