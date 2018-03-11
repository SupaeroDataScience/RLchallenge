import numpy as np
from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np
import _pickle as cPickle

#Discretisation de l'espace
def discrete_state(state):
	x = str(int(round(state['next_pipe_dist_to_player']/20)))
	y = str(int(round((state['player_y'] - state['next_pipe_bottom_y'])/20)))
	v = str(int(round(state['player_vel'])))
	return x+"-"+y+"-"+v

#GLIE actor #state = s' ajouté pour aider à la décision
def epsilon_greedy(Q, s, epsilon, state):
	a = 0

	if s in Q.keys():
		a = np.argmax(Q[s][:])
	random_act=np.random.rand()
	if random_act <= epsilon :
		if random_act <= 0.5 * epsilon:
			if state['next_pipe_bottom_y'] - state['player_y'] < 50 :
				a = 1
			else:
				a = 0
		else:
			if state['player_y'] - state['next_pipe_top_y'] > 50 :
				a = 0
			else:
				a = 1
	return a

# passer de 1 à 119
def call_action(a):
	if a==0:
		action=0
	else:
		action=119
	
	return action

#Init
gamma = 0.95
alpha = 0.9
epsilon = 0.1
nb_games = 60000
resolution = 10
Q= dict()
game = FlappyBird(graphics="fixed")
p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=True, display_screen=False)
score =0
score_100=0
# Q-learning
for i in range(1,nb_games):
	if i%100 == 0:
		print('moyenne sur 100 : %.2f' %(5+score_100 /100)) #dernière reward = -5
		if score_100/100>200:
			break
		score_100 = 0 # reset score100

	if i% 1000 == 0 :
		if alpha>0.1:
			alpha/=2
		print('parties jouées : %d, états recensés : %d' %(i,len(Q)))
		print('Moyenne : %.2f' % (5 + score / 1000)) #dernière reward = -5
		if score /1000 > 100:
			break
		score = 0 # reset score

	if i% 4000 ==0:
		epsilon/=2
	#Init du Q-learning
	p.init()
	p.reset_game()
	state=game.getGameState()
	reward = training_reward = 0

	s = discrete_state(state)
	action = epsilon_greedy(Q,s,epsilon,state)
	Q[s] = [0.0,0.0]

	while not p.game_over(): # repeat
		
		reward = p.act(call_action(action)) #retourne un entier correspondant la récompense associée à l'action 0 si action sans effet immediat, 1 si on depasse un tuyau et -5 si l'on meurt.
		if reward == -5: 
			training_reward = -1000 #rejet de cette action 
		else: 
			training_reward = 1

		state_ = game.getGameState() #s'
		s_ = discrete_state(state_)#s' discrete
		action_ = epsilon_greedy(Q,s_,epsilon, state_) #In s, choose a (GLIE actor)
		#added s' to help the action choice in obvious situation

		if s_ not in Q.keys():
			Q[s_] = [0.0,0.0]
		
		delta = (training_reward + gamma * np.max(Q[s_][:]) - Q[s][action]) #Temporal difference: δ=r+γmaxa′Q(s′,a′)−Q(s,a)
		Q[s][action]=Q[s][action] + alpha *delta #Update Q: Q(s,a)←Q(s,a)+αδ
		
		s = s_ #s←s′
		action =action_

		score +=reward
		score_100+=reward

with open('Qql', 'wb') as f:
	cPickle.dump(Q,f) 

