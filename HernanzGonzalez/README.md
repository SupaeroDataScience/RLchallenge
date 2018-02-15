## ** Read me file for my code **

In this folder you will find my version of a learning algorithm that learns to play the Flappy Bird game from PLE.

The game is played by executing the run.py file and calls the FlappyPolicy function inside the FlappyAgent.py file. This function
selects the optimal policy for the bird in any given state. 

The optimal policy is selected by refering to a dictionary (saved inside the Q.npy file). This dictionary saves values for the utilities of
each of the two possible actions of the bid (do nothing or flap). The values are saved by using a Q learning algorithm which is implemented
in the q-learning.py file. 

The bibliography used to learn about this algorithm and implement is:

https://en.wikipedia.org/wiki/Q-learning

I want to also point out several examples that have been very useful in order to better understand the way the algorithm works: 

http://sarvagyavaish.github.io/FlappyBirdRL/
