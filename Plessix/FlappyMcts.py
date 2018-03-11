"""
Class and functions used for training

"""


#%% Misc. Imports
from random import randint
import math

#%% Parameters Definition

#Jeu :

moves = [119,None]

#%% Class Definitions

#Discretization of the state returned by the game
def etat_discret(game):
    scale_dist_to_player = 4
    scale_hauteur = 8
    
    if game == None:
        player_vel = 0.0
        next_pipe_dist_to_player = 0
        hauteur = 0
    else :
        player_vel = game.getGameState()['player_vel']
        next_pipe_dist_to_player = int(game.getGameState()['next_pipe_dist_to_player']/scale_dist_to_player)
        hauteur = int((int(game.getGameState()['player_y']) -int(game.getGameState()['next_pipe_top_y']))/scale_hauteur)   
    
    return (player_vel, next_pipe_dist_to_player, hauteur)

class node:
    def __init__(self, etat_discret):
        self.play_value = 0
        self.win_value = 0.0
        self.state = etat_discret
        self.child_act_119 = () #Etat fils par l'action 119
        self.child_act_None = () #Etat fils par l'action None
        
    def __repr__(self):
        return "play : " + str(self.play_value) + '\n' \
        "win : " + str(self.win_value) + '\n' \
        "state : " + str(self.state) + '\n' \
        "119 : "+ str(self.child_act_119) + '\n' \
        "None : "+ str(self.child_act_None)
                
  
# Search for the state in the tree 
#If the node exists, the associated node is returned
#else, the function returns "None"        
def search_node(arbre, etat_discret):
    if etat_discret in arbre.keys():
        return arbre[etat_discret]
    else : 
        return None


#Returns the action to perform, using the current state and the UCB1 strategy
def selection_expansion(arbre, etat_discret):
    
    #If there is at least one unexplored node, the action is chosen randomly
    if(arbre[etat_discret].child_act_119 == () or arbre[etat_discret].child_act_None == ()):
        action = moves[randint(0,1)]

    else:    #Else : the node is chosen according to the UCB1 strategy
        child_node_119 = arbre[arbre[etat_discret].child_act_119]
        child_node_None = arbre[arbre[etat_discret].child_act_None]
        len_total = math.log(child_node_119.play_value + child_node_None.play_value)
        UCB1_value_119 = (child_node_119.win_value/child_node_119.play_value) + math.sqrt(2) * math.sqrt(len_total/child_node_119.play_value)
        UCB1_value_None = (child_node_None.win_value/child_node_None.play_value) + math.sqrt(2) * math.sqrt(len_total/child_node_None.play_value)                
        
        if (UCB1_value_119 > UCB1_value_None):
            action = 119
        else : 
            action = None
            
    return action 

#Save the built tree in a file
#ex : nom_sauvegarde = "fichier.txt"
def save_tree(arbre, nom_sauvegarde):
    with open(nom_sauvegarde, "w") as file :
        for state in arbre.keys():
            current_node = arbre[state]
            file.write(str(current_node.play_value))
            file.write('/')
            file.write(str(current_node.win_value))
            file.write('/')    
            file.write(str(current_node.state[0]))
            file.write(',')
            file.write(str(current_node.state[1]))
            file.write(',')
            file.write(str(current_node.state[2]))
            file.write('/')
            if current_node.child_act_119 == ():
                file.write('None')
                file.write('/')
            else:
                file.write(str(current_node.child_act_119[0]))
                file.write(',')
                file.write(str(current_node.child_act_119[1]))
                file.write(',')
                file.write(str(current_node.child_act_119[2]))
                file.write('/')
                
            if current_node.child_act_None == ():
                file.write('None')
            else:
                file.write(str(current_node.child_act_None[0]))
                file.write(',')
                file.write(str(current_node.child_act_None[1]))
                file.write(',')
                file.write(str(current_node.child_act_None[2]))
            file.write('\n')            


#load a pre-trained tree from a file in an empty dictionnary
def load_tree(arbre, trained_tree_address):
       
    with open(trained_tree_address,"r") as file :
        #Import data from existing file
        for line in file :
            
            extracted_list = line.rstrip('\n').split("/")
            new_play_value = int(extracted_list[0])
            new_win_value = float(extracted_list[1])
            new_state_list = extracted_list[2].split(',')
            
            new_state_player_vel = float(new_state_list[0])
            new_state_next_pipe_dist_to_player =int( new_state_list[1])
            new_state_hauteur = int(new_state_list[2])
            
            new_state = (new_state_player_vel,new_state_next_pipe_dist_to_player,new_state_hauteur)
            
            new_child_119_list = extracted_list[3]
            if new_child_119_list == 'None':
                new_child_119 = ()
            else : 
                new_child_119_list = extracted_list[3].split(',')
                new_child_119_player_vel = float(new_child_119_list[0])
                new_child_119_next_pipe_dist_to_player =int(new_child_119_list[1])
                new_child_119_hauteur = int(new_child_119_list[2])
            
            new_child_119 = (new_child_119_player_vel,new_child_119_next_pipe_dist_to_player,new_child_119_hauteur)
            
            
            new_child_None_list = extracted_list[4]
            if new_child_None_list == 'None':
                new_child_None = ()
            else : 
                new_child_None_list = extracted_list[4].split(',')
                new_child_None_player_vel = float(new_child_None_list[0])
                new_child_None_next_pipe_dist_to_player =int(new_child_None_list[1])
                new_child_None_hauteur = int(new_child_None_list[2])                    
           
            new_child_None = (new_child_None_player_vel,new_child_None_next_pipe_dist_to_player,new_child_None_hauteur)
            
            #New node creation
            new_node = node(new_state)
            new_node.play_value = new_play_value
            new_node.win_value = new_win_value
            new_node.child_act_119 = new_child_119
            new_node.child_act_None = new_child_None
            

            arbre[new_node.state] = new_node
            
    return(arbre)


    

        
