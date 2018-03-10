# Deep-Q Learning for Flappy Bird

A reinforcement learning approach to perform high scores on Flappy Bird

### Prerequisites

This project was built with:

```
Python 2.7
Tensorflow 1.4.0
```

### Installing

See description in README in folder above

## Content

* [TrainFlappy.py](TrainFlappy.py) Training of neural network implementing bird's actions
* [FlappyAgent.py](FlappyAgent.py) Creation of bird's behavior in evaluation mode
* [Run.py](Run.py) Test of our model on 100 games


### Code Explanation

This project architecture has been largely created thanks to Reinforcement Learning Notebook n°4: Deep Reinforcement Learning. Hence it includes in the same way :
```
- Raw pixels as input with 80*80 greyscale images
- Q network as learning function
- Reward clipping between 0 and 1 to facilitate learning
- Experience replay to prevent training on correlated data
```

Hyperparameters tuning has been helped by [this](https://github.com/yanpanlau/Keras-FlappyBird) article 

```
- Adam learning rate : 1e-4
- Epsilon linearly decreasing from 0.1 to 0.001 after 1 100 000 iterations
- Replay memory size : 1 000 000
- Minibatch size : 32
- Gamma : 0.99
```
Training as been done on a 1050 GTX GPU on 900 000 iterations (about 9h) leading to a 266 average score.
