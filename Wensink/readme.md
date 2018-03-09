# RL FlappyBird Project

Welcome to the FlapyBird project developed during the spring semester by Theo Wensink, as part of the Machine Learning Course at ISAE-SUPAERO.

## Getting Started

During this project, I have developed a Q-Learning algorithm enabling FlappyBird to learn learn to play.

## Code and development details

First of all, I created a script train_qvalues.py which builds the Q-Values from the experience of playing the game. Distance values are discretised using a 10x10 grid making the space of states smaller and therefore all states will be known much faster. You'll find more details in the script which has been extensively commented for a better understanding of the code. After a few hours, the Q-Values dictionary is sufficiently filled for the bird to go through a large amount of pipes.

We have then edited the FlappyAgent.py script for it to read the values from the Q-Values stored in a JSon file and decide which action is best for each state the bird goes through. We had to add the discretisation function too.

You can then simply execute the run.py and observe the magic !


## Built With

* [PyGame Learning Environment](http://pygame-learning-environment.readthedocs.io/en/latest/user/games/flappybird.html) - PyGame Learning Environment (PLE) is a learning environment, mimicking the Arcade Learning Environment interface, allowing a quick start to Reinforcement Learning in Python. 
* [Python 3.6](https://www.python.org/downloads/release/python-360/) - Python 3.6 version 

## Authors

* **Theo Wensink** - [Theo Wensink's GitHub](https://github.com/theowensink)

## Acknowledgments

* Stack Exchange Q&A
* ML Supaero course
* Friends
* Inspiration
