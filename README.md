# RL challenge

This project is a part of Supaero reinforcement learning module. The objective is to play Flappy Bird using Deep-Q Learning.
A similar project has already been done in class for a breakout game giving an example of the global method and a MemoryBuffer class.

# Method

This project uses a deep neural network to learn the Q-function which is the maximum expected reward when choosing a certain action. Knowing this Q-function defines a policy, which will the to choose the action which leads to the maximum expected reward.

This neural network takes as input raw pixels as 80*80 images. All the different states, actions and rewards are stored in a MemoryBuffer which allows to train our network on randomly chosen states (experience replay). An epsilon-greedy approach is done with epsilon starting from 0.1 and going linearly to 0.001 after 300 000 steps.

# Installation

You will need to install a few things to get started.
First, you will need PyGame.

```
pip install pygame
```

And you will need [PLE (PyGame Learning Environment)](https://github.com/ntasfi/PyGame-Learning-Environment) which is already present in this repository (the above link is only given for your information). To install it:
```
cd PyGame-Learning-Environment/
pip install -e .
```
Note that this version of FlappyBird in PLE has been slightly changed to make the challenge a bit easier: the background is turned to plain black, the bird and pipe colors are constant (red and green respectively).

# See Flappy Bird fly

The "Training.py" is the script used to generate the neural network "DQN".
To test it, execute "run.py" which will launch 10 games and store the mean score and max score for each game.
