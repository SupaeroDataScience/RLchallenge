import numpy as np
from ple.games.flappybird import FlappyBird
from ple import PLE


def FlappyPolicy(state, screen):
    # Uses a trained agent with Q-function matrix and acts Q-greedily
    # If you load with commentary "5000_epochs_multistep_initialized", the agent has been trained during 5000 epochs with multi step Q learning
    # initialized with a phase in which the 9 last actions were given a - 1000 reward
    agent = FlappyQLearner(10)
    action=None
    agent.load("4200_epochs_multistep_initialized_FINAL")
    action = agent.act(state)
    return action

class FlappyQLearner:
    # This class represent an actor critic architecture that learns to act Q-greedily with multi-step Q learning phase
    def __init__(self, max_game):
        self.epsilon = 1.0
        self.freq_decay = 10000
        self.learning_rate = 0.4
        self.gamma = 0.99  # Future is really important
        self.Q = np.random.randint(0,2,size=(14,256,36,2))  # Q matrix that represents the actor-critic learned architecture 
        self.max_game = max_game     # Number of games for training
        
    def convert_state(self, s):
        # Takes s at its raw shape from the game (dictionary of 8 elements)
        # Returns a state of 3 dimensions with reduced size (velocity, vertical distance to top of next pipe, horizontal distance to next pipe)
        new_s = []
        # Sizeof velocity state is divided by 2
        new_s.append(int((s['player_vel']+16)/2))
        # Size of height difference is divided by 4
        new_s.append(int((s['player_y']-s['next_pipe_top_y']+512)/4))
        # Size o distance to next pipe is divided by 8
        new_s.append(int(s['next_pipe_dist_to_player']/8))
        return new_s
    
    def act(self,state):
        # Acts Q-greedily
        s = self.convert_state(state)
        action_ = np.argmax(self.Q[s[0],s[1],s[2],:])
        if action_ == 0:
            action = None
        elif action_ ==1:
            action = 119
        return action
    
    def act_epsilon_greedy(self,s):
        # Acts epsilon-greedily
        alea = np.random.rand()
        if alea <= self.epsilon:
            print("NOISY ACTION")
            action_ = np.random.randint(2)
        else:
            print("Q-Greedy ACTION")
            action_ = np.argmax(self.Q[s[0],s[1],s[2],:])
        if action_ == 0:
            action = None
        elif action_ ==1:
            action = 119
        print(action)
        return action
    
    def learn(self,load,commentaire):
        # Function to perform learning of the agant with multi-step return on each game. It is possible to load a given Q matrix at the beginning
        
        # It is possible initialize learning with a numpy array already stored: if load = True we have to provide the commentary for file name
        if load:
            self.load(commentaire)
        count = np.zeros(np.shape(self.Q)) # to register the explored (s,a)
        game = FlappyBird()
        p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=True, display_screen=False)
        p.init()
        # Initialize scores and loss history to plot it at the end of learning phase
        loss = []
        scores = []
        for i in range(self.max_game):
            print("Game number: {}".format(i))
            p.reset_game()
            state = game.getGameState()
            memory = []
            last_reward = 0
            while not p.game_over():
                s = self.convert_state(state)
                a = self.act(state)
                reward = p.act(a)
                if reward > 0:
                    last_reward+= reward
                r = 5*reward+1
                next_state = game.getGameState()
                s_ = self.convert_state(next_state)
                # Convert action to be stored in shape (0,1)
                if a == 119:
                    a = 1
                else:
                    a = 0
                # We feed memory because update of Q is at the end of game
                memory.append((s,a,r,s_))
                state = next_state
            # Score of this game is the number of pipes that was crossed, that is to say the sum of all strictly positive rewards
            j = 0
            print("Score: {}".format((last_reward)))
            scores.append(last_reward)
            
            # The memory buffer allows to implement a multi-step return to accelerate and stabilize convergence
            
            # We need a reward vector
            R = [m[2] for m in memory]
            # We penalyse the last action which led to defeat. The previous actions which are also responsible will be penalized by multi-step return
            R[len(memory)-1] = - 50
            last_s_ = memory[len(memory)-1][3]
            for s,a,r,s_ in memory:
                # Creation of multi-step target
                gamma = [self.gamma**k for k in range(len(memory)-j)]
                multi_reward = R[j:len(memory)]
                target = np.sum(np.array(gamma) * np.array(multi_reward))+ self.gamma**len(memory)*np.max(self.Q[last_s_[0],last_s_[1],last_s_[2],:])
                
                # Q-update with multi-step return on visited states during the episode
                self.Q[s[0],s[1],s[2],a] += self.learning_rate * (target-self.Q[s[0],s[1],s[2],a])
                
                # We save loss
                loss.append(target-self.Q[s[0],s[1],s[2],a])
                j += 1
                # We count exploration
                count[s[0],s[1],s[2],a] += 1
        return self.Q, loss, scores, count
    
    def play(self,nb_games):
        # Function to see agent act in Flappy Bird environment
        
        game = FlappyBird()
        p = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=False, display_screen=True)
        # Note: if you want to see you agent act in real time, set force_fps to False. But don't use this setting for learning, just for display purposes.

        p.init()
        reward = 0.0
        cumulated = np.zeros((nb_games))

        for i in range(nb_games):
            p.reset_game()
            state = game.getGameState()
            print(state)
            while(not p.game_over()):
                state = game.getGameState()
                #screen = p.getScreenRGB()        
                action=self.act(state) ### Act Q-greedily
                reward = p.act(action)
                cumulated[i] = cumulated[i] + reward

        average_score = np.mean(cumulated)
        max_score = np.max(cumulated)
        return average_score, max_score
    
    def save(self,commentaire):
        # Saves the Q matrix in folder "data" with name in parameter
        np.save("data/Q_"+commentaire,self.Q)
        
    def load(self,commentaire):
        #loads Q matrix in atributes Q of agent with name in parameter
        self.Q = np.load("data/Q_"+commentaire+".npy")



