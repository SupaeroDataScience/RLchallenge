import numpy as np
import pickle
from ple import PLE
from ple.games.flappybird import FlappyBird
from state import new_state

#Retourner l'action en fonction du argmax (0 ou 1)
def get_action(a):
    return a*119

#Def epsilon greedy
def epsilon_greedy(Q, new_state, epsilon, state):
    a = np.argmax(Q[new_state[0],new_state[1],new_state[2]])
    if np.random.rand() <= epsilon :
        if np.random.rand() <= 0.5 * epsilon:
            if state['next_pipe_bottom_y'] - state['player_y'] < 50 :
                a = 1
            else :
                a = 0

    return a


# Parametres
gamma = 0.95
alpha = 0.9
epsilon = 0.1
nb_games = 15000

#taille de notre espace des états
X = 18
Y = 30
V = 21

# Init Q
Q = np.zeros((X,Y,V,2))
#file = open("Qtrained",'rb')
#Q = pickle.load(file)
#alpha = 0.1

#Création du jeu accéléré
game = FlappyBird(graphics="fancy")
p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=True, display_screen=False)

# Score des X dernières parties
last_100 = 0
last_1000 = 0

#calcul de l'espace des états
#Xmax = 0
#Ymax = 0
#Vmax = 0

#file=open('Qtrained', 'rb')
#Q=marshal.load(file)
# For each game
for g in range(1, nb_games):

    # Début du jeu
    p.init()
    p.reset_game()
    state = game.getGameState()
    reward = training_reward = 0
    s = new_state(state)
    action = epsilon_greedy(Q, s, epsilon, state)
    
    #calcul de l'espace des états
    #if s[0] > Xmax:
    #    Xmax = s[0]
    #if s[1] > Ymax:
    #    Ymax = s[1]
    #if s[2] > Vmax:
    #    Vmax = s[2]

    while not p.game_over():

        # Action
        reward = p.act(get_action(action))

        # Calcul de la reward d'entrainement
        if reward == -5 :
            training_reward = -100
        else:
            training_reward = 1

        # Nouvel état
        state_ = game.getGameState()
        s_ = new_state(state_)
        action_ = epsilon_greedy(Q, s_, epsilon, state)

        # calcul de Q avec l'algorythme SARSA
        delta = (training_reward + gamma * Q[s_[0],s_[1],s_[2]][action_] - Q[s[0],s[1],s[2]][action])
        Q[s[0],s[1],s[2]][action] = Q[s[0],s[1],s[2]][action] + alpha * delta

        # Update de l'état
        s = s_
        action = action_
        
        # Calcul des résultats en cours de compilation
        if reward+5:
            last_100 += reward
            last_1000 += reward
        
        # contrôle des résultats en cours de compilation et diminution de alpha
    if g %100 == 0 :
        print('Moyenne des 100 derniers essais : %.2f' %(last_100/100))
        last_100 = 0
    if g %1000 == 0 :
        while alpha > 0.1 :
            alpha /= 1.01
        print('Moyenne des 1000 derniers essais : %2f' % (last_1000/1000))
        if last_1000 / 1000 > 50:
            break
        last_1000 = 0


#Résultat de la taille de l'espace des états
#print(Xmax,Ymax,Vmax)
            
#Sauvegarde des données avec pickle, marshal ne marchant pas
with open('Qtrained', 'wb') as f:
    pickle.dump(Q,f)