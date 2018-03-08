# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 18:37:50 2018

@author: Carlos CASTA
"""

from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np
import random 



game = FlappyBird(graphics="fixed") # use "fancy" for full background, random bird color and random pipe color, use "fixed" (default) for black background and constant bird and pipe colors.
p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=True, display_screen=True)
# Note: if you want to see you agent act in real time, set force_fps to False. But don't use this setting for learning, just for display purposes.

#PARAMETROS
alpha = 0.1
gamma = 0.9
epsilon = 0.1
actions = [0,119]


#DEFINIMOS UN DICCIONARIO 
matrizQ = dict()
#¥matrizQ={0,0}
#matrizQ = np.load("QMATRIX.npy").item()

#INICIAMOS EL JUEGO
p.init()
reward = 0.0
nb_games = 1000
contador1 = 0
contador2 = 0
cumulated = np.zeros((nb_games))


for i in range(nb_games):
    p.reset_game()
    
    contador1 = contador1 + 1

    # Print game data
    if(contador1 % 100 == 0): # to reduce the value of epsilon 
        epsilon *= 0.8
        
        average_score = np.mean(cumulated[contador1-100:contador1-1])
        max_score = np.max(cumulated[(contador1-100):contador1-1])
        
        print(max_score, average_score, contador1, contador2, epsilon, flush=True)

    
    while(not p.game_over()):
        state = game.getGameState()
        
        disty=state["next_pipe_bottom_y"]-state["player_y"]
        distx= state["next_pipe_dist_to_player"]
        velp=state["player_vel"]
        
        if distx < 3000:
            distx = int(distx) - (int(distx) % 20)
        else:
            distx = 3000 

        if disty < 1800:
            disty = int(disty) - (int(disty) % 20)
        else:
            disty = 1800
            
        if velp < 100:
            velp = int(velp) - (int(velp) % 2)
        else:
            velp = 100
    
        
        stringdisty= str(disty)
        stringdistx= str(distx)
        stringvelp= str(velp)
        
        statep= "".join ([stringdisty," / ", stringdistx," / ", stringvelp])
        #print(statep)
        
        
        if (matrizQ.get(statep) == None):
            matrizQ[statep] = [0 , 0]
       
        contador2 = contador2 + 1
        
        #if(contador1 % 100 == 0): # to reduce the value of epsilon 
        #    epsilon = epsilon * 0.997
        
        
        if (epsilon < random.random()):
            #action=random.randint(0,1)*119
            bestindex = np.argmax(matrizQ[statep])
            action = actions[bestindex]
    
        else:
            bestindex = random.randint(0,1)
            action = bestindex * 119
            
        reward = p.act(action)
        cumulated[i] = cumulated[i] + reward

        
        
         # Miramos el siguiente estado y le asignamos una llave
        state = game.getGameState()
        disty2=state["next_pipe_bottom_y"]-state["player_y"]
        distx2 = state["next_pipe_dist_to_player"]
        velp2=state["player_vel"]
        
        if distx2 < 3000:
            distx2 = int(distx2) - (int(distx2) % 20)
        else:
            distx2 = 3000 

        if disty2 < 1800:
            disty2 = int(disty2) - (int(disty2) % 20)
        else:
            disty2 = 1800
        
         if velp2 < 100:
            velp2 = int(velp2) - (int(velp2) % 2)
        else:
            velp2 = 100
    
        
        stringdisty2= str(disty2)
        stringdistx2= str(distx2)
        stringvelp2= str(velp2)
        
        statep2= "".join ([stringdisty2," / ", stringdistx2," / ", stringvelp2])
        
        if (matrizQ.get(statep2) == None):
            matrizQ[statep2] = [0 , 0]
        
        # actualiazamos el valor de Q
        matrizQ[statep][bestindex] = matrizQ[statep][bestindex] + alpha*(reward + gamma*max(matrizQ[statep2])-matrizQ[statep][bestindex])
        
    
    
np.save("QM.npy",matrizQ)