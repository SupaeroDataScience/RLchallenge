# Presentation

This repository is the result of a school Reinforcment Learning Challenge aiming to train an policy for FlappyBird atari game.
Two methods have been deployed to do so : 

# Computation constraints : 
All calculations and training phases have pushed to Gcloud platform with a free trial account : (8vCPUS, 30GB Memory)
To do so, the line allows the game not to display any window during the training phase (which would generate an error since Gcloud Virtual Machines come without UI) : 

os.environ['SDL_VIDEODRIVER'] = 'dummy'

# QLearning
Largely inspired from previous Lessons & Fellow Students results, the main idea of this QLearning is to crop the state-space and to reward  
# DQN
Training
