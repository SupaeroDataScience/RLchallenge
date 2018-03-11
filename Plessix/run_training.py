# -*- coding: utf-8 -*-
"""
Parameters definition to perform the training of a MCTS.

The tree can be created from scratch or can be loaded from
an existing .txt file.

@author: guillaume
"""
#%% Import needed class and functions

import FlappyMcts

#%% Import game libraries
from ple.games.flappybird import FlappyBird
from ple import PLE


#%% Training Parameters Definition

forceFPS = True

#Save training history (every 10 games) in 'apprentissage' file:
    # training step
    # number of discovered nodes
    # current average score
apprentissage = 'apprentissage' #Define history location

#Define number of trainings to perform
games = 10

#Define death penalty
penalite = -5.0

#Define number of penalised states (backpropagation)
etats_penalises = 9

#The tree is saved in a file
nom_sauvegarde = 'toto'

#A pre-trained tree can be loaded from a file :
trained_tree_address = 'trained_tree'
#trained_tree_address = None #If there is no tree to load

#%% Training Function
             
#The tree is built by exploring the different states and their child nodes 
#Inputs : 
        # trained_tree_address :  name of the pre-trained tree file. If "None", we start from scratch
        # nom_sauvegarde : file in which the tree is saved after training (ex : "trained_tree.txt")
#                Note : a new file is created every 100 trainings to avoid losing data
        
#Return :
        # Trained Tree (as a dictionnary)
def MCTS_training(trained_tree_address, nom_sauvegarde):
    #Tree initialization
    if trained_tree_address == None :     
        #An empty tree is built
        main_tree = {}
    else :
        main_tree = {}
        main_tree = FlappyMcts.load_tree(main_tree, trained_tree_address)
    
    #Game Initialization
    game = FlappyBird()

    p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=forceFPS, display_screen=True)
    p.init()
    score = 0
    
    for i in range(games):
        
        #Restart game
        p.reset_game()
        current_state = FlappyMcts.etat_discret(game)
        bar_passed = 0
        
        
        while(not p.game_over()): #While the game is not over
            visited_states = []
            reward = 0.0
                
            current_state = FlappyMcts.etat_discret(game)
            #If the nodes exists, it is found in the tree
            #Else, the function return "None"
            current_node = FlappyMcts.search_node(main_tree, current_state)
                            
            if current_node == None : #The node does not exists in the tree : it is the first time the state is visited
                current_node = FlappyMcts.node(current_state) #A new node is created
                main_tree[current_state] =  current_node #The node is added to the tree

                
            main_tree[current_state].play_value += 1 #Update of the play value of the node
            visited_states.append(current_state)
            while(reward == 0.0): #The game is not over and no pipe is passed
            
                #Action is chosen based on UCB1 method
                action = FlappyMcts.selection_expansion(main_tree, current_state)
                
                #Perform chosen action
                reward = p.act(action)
                
                next_state = FlappyMcts.etat_discret(game)
                
                #If the nodes exists, it is found in the tree
                next_node = FlappyMcts.search_node(main_tree, next_state)
                                
                if next_node == None : #The node does not exists in the tree : it is the first time the state is visited
                    next_node = FlappyMcts.node(next_state) #A new node is created
                    main_tree[next_state] = next_node #The node is added to the tree
                
                main_tree[next_state].play_value += 1 #Update of the play value of the node
                visited_states.append(next_state)
                
                #Update of the node's child, depending on the action
                if(action == None):
                    main_tree[current_state].child_act_None = next_state
                else :
                    main_tree[current_state].child_act_119 = next_state

                current_node = next_node
                current_state = next_state
                
            if reward > 0.0 : #A pipe is passed
                #Count of the total passed pipes
                bar_passed += 1
                #The nodes that lead to pass the pipes are rewarded
                for state in visited_states:
                    main_tree[state].win_value += 1.0
                
                
            #A final state is reached (end of the game)
            if reward < 0.0 : 
                #Backpropagation of the penalty on the last nodes
                for j in range(min(len(visited_states),etats_penalises)):
                    selected_state = visited_states.pop()
                    main_tree[selected_state].win_value += penalite        
        score += bar_passed            
                    
        if i%10 == 0 :
            print(i)
            with open(apprentissage, "a") as f :
                f.write('step :' + str(i) + '/'+'graphsize:'+str(len(main_tree))+'/Score:' + str(score/10) + '\n')
            score = 0    
                
        if i%100 == 0 :
            FlappyMcts.save_tree(main_tree, nom_sauvegarde + str(i))

    return main_tree

#%% Perform training

MCTS_training(trained_tree_address, nom_sauvegarde)
