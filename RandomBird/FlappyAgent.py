# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 14:41:52 2018

@author: Arnaud
"""
from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np
import pickle

def epsilon_greedy(Q, y, dist,veloc, epsilon):
    a = np.argmax(Q[y,dist,veloc,:])
    if (a == 0) : 
        a=None
    else:
        a=119
    if(np.random.rand()<=epsilon): # random action
        aa = np.random.randint(0,2)
        if (aa == 0) : 
            a=None
        else:
            a=119
            
    return a



# Stockage des 9 dernières paires états-actions en cas de défaite
def updateLast9(last9,y,dist,veloc,ia):
    for i in range(0,9):
        last9[i]=last9[i+1]
    last9[9][0] = y
    last9[9][1] = dist
    last9[9][2] = veloc
    last9[9][3] = ia

nb_games = 1000000
sizey = 15 #-300 à 300 #Différence entre le haut de next_pipe et l'oiseau 
sizex = 10 # 0 à 283 #distance entre l'oiseau et next_pipe
sizev = 9 #vitesse discretisée


#Q-learning, matrice état + actions
#Qql = np.zeros((sizey,sizex,sizev,2))
epsilon = 0 #L'aléatoire n'est pas nécessaire, au contraire il réduit les performances car en pénalisant 9 pairs états-actions on risque de pénaliser N pairs intéressantes pour 9-N pairs aléatoires qui ont mené l'oiseau à sa perte

r=0  #initialisation de la récompense


#Parametres du modèle
alpha = 0.05# Des valeurs de alpha de l'ordre de 0.4 étaient intéressante pour gagner des performances rapidement mais menaient systématiquement à du sur-apprentissage et à des pertes de performances (ex : montée à 80 de moyenne puis chute à 15 puis stagnation à 15)
gamma = 0.95

#Initialisation des états
y = 0
dist = 0
veloc = 0
ia = 0
cumulated = np.zeros((nb_games))+5
#
## Read from file
f_myfile = open('Q_functionArnaud.pickle', 'rb')
Qql = pickle.load(f_myfile)  
f_myfile.close()


def FlappyPolicy(state, screen):

    y=int((state['player_y']-state['next_pipe_bottom_y']+300)/40)
    dist = int(state['next_pipe_dist_to_player']/40)
    veloc = int((state['player_vel']+16)/3)
    a = np.argmax(Qql[y,dist,veloc,:])
   
    if (a == 0) : 
        a=None
    else:
        a=119 

    return a
##ENTRAINEMENT
#train = True pour lancer l'entrainement
train = False
if train:
    game = FlappyBird()
    p = PLE(game,fps=30, frame_skip=1, num_steps=1, force_fps=True, display_screen=True)
    p.init()

    for i in range(0,nb_games):
        p.reset_game()
        state = game.getGameState()
        screen = p.getScreenRGB()
        y=int((state['player_y']-state['next_pipe_bottom_y']+300)/40)
        dist = int(state['next_pipe_dist_to_player']/40)
        veloc = int((state['player_vel']+16)/3)
        last9 = np.zeros((10,4))
        while(not p.game_over()):
            
        
            if((i+1)%100==0):
                epsilon = epsilon/2
            if ((i+1)>300):
                epsilon = 0
            
            a = epsilon_greedy(Qql,y,dist,veloc,epsilon)
            if (a == None):
                ia = 0
            else:
                ia = 1
            
            action=a ### Your job is to define this function.
            rewGame = p.act(action)
            state = game.getGameState()
            screen = p.getScreenRGB()
            ##Update de l'etat
            yprec = y
            distprec = dist
            velocprec = veloc
            y=int((state['player_y']-state['next_pipe_bottom_y']+300)/40)
            dist = int(state['next_pipe_dist_to_player']/40)
            veloc = int((state['player_vel']+16)/3)
            
            
            r=1
            updateLast9(last9,y,dist,veloc,ia)
            Qql[yprec][distprec][velocprec][ia] = Qql[yprec][distprec][velocprec][ia] + alpha * (r+gamma*np.max(Qql[y][dist][veloc][:])-Qql[yprec][distprec][velocprec][ia])
          
           
            cumulated[i] = cumulated[i] + rewGame
            
        for l in range(0,8):
            r=-1000
            yprec = int(last9[l][0])
            y = int(last9[l+1][0])
            distprec = int(last9[l][1])
            dist = int(last9[l+1][1])
            velocprec = int(last9[l][2])
            veloc = int(last9[l+1][2])
            ia = int(last9[l+1][3])
            Qql[yprec][distprec][velocprec][ia] = Qql[yprec][distprec][velocprec][ia] + alpha * (r+gamma*np.max(Qql[y][dist][veloc][:])-Qql[yprec][distprec][velocprec][ia])
        
        if (i<100):
            print("i = ",i," - cumulated[i] = ",cumulated[i]," - mean = ",np.mean(cumulated[0:i]))
        else:
            print("i = ",i," - cumulated[i] = ",cumulated[i]," - mean[-1000] = ",np.mean(cumulated[i-100:i]))
            
average_score = np.mean(cumulated[20000:24055])
max_score = np.max(cumulated)
#
#print("saving model")
#f_myfile = open('Q_functionArnaud.pickle', 'wb')
#pickle.dump(Qql, f_myfile)
#f_myfile.close()

