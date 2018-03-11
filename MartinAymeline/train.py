### Imports

# Import ple games library
from ple.games.flappybird import FlappyBird
from ple import PLE
# Import common python tools
import numpy as np
import time
from collections import deque
# Import neural network tools
from keras.models import load_model
# Import built tool functions to train
import utilities
from replay_memory import MemoryBuffer
# Import constantes
from constantes import constantes as cst

### Main training function
    
def programme(training):
    
    ## STARTING PART
    
    if training == "init" :
        # Create the network
        dqn = utilities.create_network()
        print("New created network")
        name = 'model_dqn_new.h5'
    else : 
        # Load an existing one
        dqn = load_model('model_dqn_to_train.h5')
        print("Existing model load")
        name = 'model_dqn_to_train.h5'
        
    input("Continue ?")
            
    # Start Flappy game and the environment
    game = FlappyBird(graphics="fixed")
    env = PLE(game, fps=30, frame_skip=1, num_steps=1, force_fps=True, \
              display_screen=True)
    possible_actions = env.getActionSet() # return [119, None]
    # Initialize the environment and the key indicators
    env.init() 
    reward = 0.0
    loss = 0.0

    ## INITIALIZATION PART
    
    # Start a new game
    env.reset_game()
    # Note : env.act(possible_actions[0]) <-> action "119" <-> GO UP
    #        env.act(possible_actions[1]) <-> action "None" <-> DO NOTHING

    # Initialize the "state" which is here the screen of the game
    screen_x = utilities.process_screen(env.getScreenRGB()) 
    # We stack 4 last screen images to take speed into account in the trainning
    stacked_x = deque([screen_x, screen_x, screen_x, screen_x], maxlen=4)
    x = np.stack(stacked_x, axis=-1)
    # Initialize the memory buffer which will be used to replay experience
    replay_memory = MemoryBuffer(cst.replay_memory_size, screen_x.shape, (1,))

    # Store the initial state for further evaluations
    Xtest = np.array([x])
    # Initialize evaluation indicators
    scoreQ = np.zeros((cst.nb_epochs))
    scoreMC = np.zeros((cst.nb_epochs))
    scoreMax = np.zeros((cst.nb_epochs))
    # Initialize timer
    start = time.time()
    
    ## TRAINING PART

    # Here is a deep-q-learning method with experience replay
    for step in range(cst.total_steps):
        
        # EVALUATION :
        
        # We evaluate the network performances every 5000 steps
        if(step % cst.evaluation_period == 0 and step > 0):
            cst.epoch += 1
            print('[Epoch {:d}/{:d}] {:d} steps done'.format(cst.epoch+1, \
                  cst.total_steps//cst.evaluation_period, cst.evaluation_period))
            # Evaluation on the initial state
            scoreQ[cst.epoch] = np.mean(dqn.predict(Xtest).max(1))
            # Roll_out evaluation : we store mean and max scores, at each 
            # evaluation step, over 20 games 
            scoreMC[cst.epoch], scoreMax[cst.epoch] = utilities.MCeval(env, 20, \
                   dqn, cst.gamma)
            # We save the evaluated network
            dqn.save(name)
            # And the evaluated scores
            with open('eval.log','a') as f:
                f.write(str(cst.epoch)+','+str(scoreQ[cst.epoch])+','+ \
                        str(scoreMC[cst.epoch])+','+str(scoreMax[cst.epoch])+'\n')

        # PLAY :
        
        # Action selection : a random float is computed in [0,1]. Then the action
        # is chosen randomly if the float is lower than our annealing epsilon, 
        # otherwise the action is chosen using the current network.
        if np.random.rand() < utilities.epsilon(step):
            print("Random action")
            # When a random action is selected, the following formula decides the
            # action. We fix it so that there is a 12,5% chance that the chosen
            # action will be to go up (a=0 <--> action 119).
            a = 1 - np.random.randint(len(possible_actions))*np.random.randint(len(possible_actions))*np.random.randint(len(possible_actions))
        else:
            print("Greedy action")
            # Otherwise, the action is chosen by the dqn.
            a = utilities.greedy_action(dqn, x)
        
        # The chosen action is played
        r = env.act(possible_actions[a])
        # We determine the reward for this action and its result
        reward = utilities.clip_reward(r)
        screen_y = utilities.process_screen(env.getScreenRGB())
        # Then we add the step in the memory buffer
        replay_memory.append(screen_x, a, reward, screen_y, env.game_over())
    
        # NETWORK LEARNING :
        
        if step > cst.mini_batch_size and step > cst.observation:
            # After an exploring phase we start training the network
            X,A,R,Y,D = replay_memory.minibatch(cst.mini_batch_size)
            QY = dqn.predict(Y)
            QYmax = QY.max(1).reshape((cst.mini_batch_size,1))
            update = R + cst.gamma * (1-D) * QYmax
            QX = dqn.predict(X)
            QX[np.arange(cst.mini_batch_size), A.ravel()] = update.ravel()
            loss += dqn.train_on_batch(x=X, y=QX)
        
        # NEXT STEP PREPARATION :
        
        if env.game_over()==True:
            # The episode is restarted if the game is over at this step
            env.reset_game()
            screen_x = utilities.process_screen(env.getScreenRGB())
            stacked_x = deque([screen_x, screen_x, screen_x, screen_x], maxlen=4)
            x = np.stack(stacked_x, axis=-1)
        else:
            # Otherwise, the game keep going
            screen_x = screen_y
            stacked_x.append(screen_x)
            x = np.stack(stacked_x, axis=-1)
        
        if step > cst.observation :
            print("STEP", step, ": Epsilon is ", utilities.epsilon(step), \
              ", the chosen action is", possible_actions[a], ". The reward", r ,"or", reward, \
             "and the Loss is", loss)
        else :
            print("STEP", step, ": Epsilon is", utilities.epsilon(step), \
              ", the chosen ation is", possible_actions[a], "and the reward is", r ,"or", reward)
    
    # After the last step we save the trained network
    dqn.save('model_dqn_to_train.h5')
    
    print("End of training in {:d} seconds !".format(time.time() - start))


# Main to launch the training by chosing if we keep training an existing network
# or if we start training a new one
if __name__ == "__main__":
    
    training_step_choice = input("""Write "init" if you start training the CNN or "keep_going" otherwise """)
    programme(training_step_choice)
    
    