# RL challenge

My challenge is to learn to play [Flappy Bird](https://en.wikipedia.org/wiki/Flappy_Bird)!

Flappybird is a side-scrolling game where the agent must successfully nagivate through gaps between pipes. Only two actions in this game: at each time step, either you click and the bird flaps, or you don't click and gravity plays its role.



# DEEP Q-LEARNING

For this project, I decided to use deep Q-learning. For that, I used Keras and Tensorflow libraries. I worked on the state vector and created a neural network to learn the Q-function.
The Neural Network is only one layer. But we already know that this is theoretically enough to reproduce any mathematical function.

A lot of the project time has been spent on choosing the right optimizers, loss functions and activation functions. These seem to be very important to build a good neural network.

Moreover, the reward, although it sounded at first counter-intuitive, but it played a very big role on the quality of the learning. a very big difference in rewards leads to much better results.

# RESULTS

The model I kept, is a simple one-layer model, with no experience replay. It hits regularly 30 average score and can peak up to 190 of score.


