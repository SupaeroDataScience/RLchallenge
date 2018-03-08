## **Q-Learning algorithm to play Flappy Bird**

In this folder you will find my version of a learning algorithm that learns to play the [Flappy Bird](https://en.wikipedia.org/wiki/Flappy_Bird) game from [PLE (PyGame Learning Environment)](https://github.com/ntasfi/PyGame-Learning-Environment).

The game is played by executing the [run.py](run.py) file and calls the `FlappyPolicy(state, screen)` function inside the [FlappyAgent.py](FlappyAgent.py) file. This function selects the optimal policy for the bird in any given state. 

The optimal policy is selected by refering to a dictionary (saved inside the Q.npy file). This dictionary saves values for the utilities of each of the two possible actions of the bird (which are do nothing or flap). The values are saved by using a Q learning algorithm which is implemented in the [q-learning.py](q-learning.py) file. 

The **bibliography** used to learn about this algorithm and implement is:

- https://en.wikipedia.org/wiki/Q-learning Wikipedia is always useful for a broad and not-so-broad approach.
- http://mnemstudio.org/path-finding-q-learning-tutorial.htm A simple example of q-learning to exit a house from any of its five rooms.
- https://studywolf.wordpress.com/2012/11/25/reinforcement-learning-q-learning-and-exploration/ Another example with mouse searching for cheese on a simple grid.

I want to also point out several examples that have been very useful in order to better understand the way the algorithm works: 

- https://github.com/chncyhn/flappybird-qlearning-bot
- http://sarvagyavaish.github.io/FlappyBirdRL/

I must also say that I have been trying to implement a **convolutional neural network** that uses either the pixels of the image but I have not been successful. My attempts are visible on the [nn-learning.py](nn-learning.py). I also created a [FlappyAgent2.py](FlappyAgent2.py) file that chooses the policy using the neural network that has been trained on on the [nn-learning.py](nn-learning.py) file. It did not converge so the working version is the q-learning algorithm which creates a dictionary in [q-learning.py](q-learning.py). 

Still, there is a large amount of examples online and a great deal of tutorials and helpful documents. I point out the ones that I had been using (I think they both are great, I wish I had dedicated more time to fully understand it and successfully implement it):

- https://github.com/yenchenlin/DeepLearningFlappyBird
- https://yanpanlau.github.io/2016/07/10/FlappyBird-Keras.html
