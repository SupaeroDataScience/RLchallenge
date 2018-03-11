"""--------------------------------"""
""" Initialisation de Flappy Bird  """
"""--------------------------------"""

from ple.games.flappybird import FlappyBird
from ple import PLE

# Définition des actions
actions = [None, 119]

game = FlappyBird(graphics="fixed")
p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=True, display_screen=True)
p.init()


"""----------------------------"""
""" Création du Deep Q-Network """
"""----------------------------"""

from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

dqn = Sequential()
# 1st layer
dqn.add(Dense(units=500, kernel_initializer='lecun_uniform', activation="relu", input_dim = 8))
# output layer
dqn.add(Dense(units=2, kernel_initializer='lecun_uniform', activation="linear"))

dqn.compile(loss='mse', optimizer=optimizers.Adam(1e-4))

dqn.load_weights("dqn_2_1.dqf")
#dqn.load_weights("dqn_0_3.dqf")

"""-----------------------------------------"""
""" Définition de quelques fonctions utiles """
"""-----------------------------------------"""

import numpy as np

def process_state(state):
    """ Renvoie l'état sous forme de liste """
    return [state['player_y'], state['player_vel'],
            state['next_pipe_dist_to_player'], state['next_pipe_top_y'], state['next_pipe_bottom_y'],
            state['next_next_pipe_dist_to_player'], state['next_next_pipe_top_y'], state['next_next_pipe_bottom_y']]
    
def epsilon(step):
    """ Utile à la décision de l'action """
    #if step<1e6:
     #   return 1.-step*9e-7
    #return .1
    return 0.01

def clip_reward(r):
    """ Change la valeur de reward """
    rr=0
    if r > 0:
        rr = 1
    if r < 0:
        rr = -1000
    return rr

def greedy_action(network, state_x):
    """ Renvoie la meilleure action possible """
    Q = network.predict(np.array(state_x).reshape(1, len(state_x)), batch_size=batchSize)
    return np.argmax(Q)

def MCeval(network, games, gamma):
    """ Evaluation du réseau de neurones """
    scores = np.zeros(games)
    for i in range(games):
        p.reset_game()
        state_x = process_state(game.getGameState())
        step = -1
        while not game.game_over():
            step += 1
            action = greedy_action(network, state_x)
            reward = p.act(actions[action])
            state_y = process_state(game.getGameState())
            scores[i] = scores[i] + reward
            state_x = state_y
    return np.mean(scores)


"""----------------------"""
""" Apprentissage du DQN """
"""----------------------"""

# Variables utiles
total_games = 10000
gamma = 0.99
step = 0
batchSize = 256

# Définition des évaluations
evaluation_period = 300
nb_epochs = total_games // evaluation_period
epoch=-1
scoreMC = np.zeros((nb_epochs))

# Enregistrement du réseau de neurones
filename = "dqn_3_"


"""-----------------"""
""" Deep Q-Learning """
"""-----------------"""

for id_game in range(total_games):
    if id_game % evaluation_period == 0:
        epoch += 1
        scoreMC[epoch] = MCeval(dqn, 50, gamma)
        dqn.save(filename + str(epoch) + ".dqf")
        print(">>> Eval n°%d | score = %f" % (epoch, scoreMC[epoch]))
    p.reset_game()      # Nouvelle partie
    state_x = process_state(game.getGameState())
    id_frame = 0
    score = 0
    alea = 0
    while not game.game_over():
        id_frame += 1
        step += 1
        ## Choisit l'action à effectuer : 0 ou 1
        if np.random.rand() < epsilon(step):    # Action au hasard
            alea += 1
            action = np.random.choice([0, 1])
        else:                                   # Meilleure action possible
            action = greedy_action(dqn, state_x)
        ## Joue l'action et observe le gain et l'état suivant
        reward = p.act(actions[action])
        reward = clip_reward(reward)
        state_y = process_state(game.getGameState())
        ## Mise à jour de Q
        QX = dqn.predict(np.array(state_x).reshape(1, len(state_x)), batch_size=batchSize)
        y = np.zeros(2)
        y[:] = QX[:]
        if not game.game_over():
            score += reward
            QY = dqn.predict(np.array(state_y).reshape(1, len(state_y)), batch_size=batchSize)
            QYmax = np.max(QY)
            update = reward + gamma * QYmax
        else:
            update = reward
        y[action] = update
        dqn.fit(np.array(state_x).reshape(1, len(state_x)), np.array(y).reshape(1, len(y)), nb_epoch = 3, verbose = 0)
        state_x = state_y
    print(">>> game n°%d | score = %d | nb_steps = %d | %% aléa = %f%%" % (id_game, score, id_frame, alea/id_frame*100))

for i in nb_epochs:
    print("epoch n°%d | score = %f" % (i, scoreMC[i]))