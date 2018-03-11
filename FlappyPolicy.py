import numpy as np
from ple.games.flappybird import FlappyBird
from ple import PLE
actions = [None, 119]



def getstate(state):
    
    a = list(state.values())
    v_pos = np.int(np.round((a[0]-a[3]+512)/10)) # Différence de position verticale entre l'oiseau et le bas du prochain tuyau
                          #+512 (hauteur totale en pixels) pour ne jamais être négatif
    h_pos = np.int(np.round((a[2]+20)/10)) #Distance de l'oiseau au prochain tuyau
    
    speed = np.int(np.round(a[1]))
    S = 'v'+np.str(v_pos)+'h'+np.str(h_pos)+'s'+np.str(speed) #Transformation dans le formalisme string utilisé précédemment
    return S

#Facile de déterminer FlappyPolicy : il suffit d'aller chercher l'info dans la matrice Q
def FlappyPolicy(state, screen):
    Q = np.load('Q.npy').item()    
    S = getstate(state)
    if S not in Q:
        action = None
    elif Q[S][119]>Q[S][None]:
        action = 119
    else:
        action = None
        
    return action
