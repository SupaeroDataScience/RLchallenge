# Presentation

This repository is the result of a school Reinforcment Learning Challenge aiming to train an policy for FlappyBird atari game.
Two methods have been deployed to do so : 

# Computation constraints : 
All calculations and training phases have pushed to Gcloud platform with a free trial account : (8vCPUS, 30GB Memory)
To do so, the line allows the game not to display any window during the training phase (which would generate an error since Gcloud Virtual Machines come without UI) : 

os.environ['SDL_VIDEODRIVER'] = 'dummy'

# QLearning
Largely inspired from previous Lessons & Fellow Students results, the main idea of this QLearning is to narrowly **crop the state-space** and to punish with a large **negative** reward when flappy crashes.

# DQL
Deep Q Learning is the first solution I tried to implement (then switched to QLearning in order to present some results).
The original frame is croped to the floor and just behind the back of flappy, resized to a 84x84 grasycale window then stacked with the 3 previous states so the network as the ability to get differential information (velocity).
## Training Period : 
300 000 steps
Network saved after 50 000 steps
Policy evaluated after 25 000 steps

Network has been trained over 300 000 steps so far. 
Results and evolution tend to make me think that a longer training period would hopefully largely improve results.


