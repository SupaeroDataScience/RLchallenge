# Approach

This project implements a reinforcement learning algorithm applied to the following game: FlappyBird.
The driving idea was to implement an algorithm working directly on the raw pixels of the board. Liberally inspired from the original paper from DeepMind [link](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf), we use a deep q network to extract the features from the frames. Then we feed a fully connected layer in order to predict the probability to take each action.

The results are quite good in general, but the standard deviation is important too. On the following training (500 000 steps), we did some evaluations (on 15 games) every 20 000 steps to observe the learning curve of the algorithm.

* 20kth Iteration - Avg: -4.86 - Max: -4.0
* 40kth Iteration - Avg: -4.53 - Max: -3.0
* 60kth Iteration - Avg: -4.6 - Max: -3.0
* 80kth Iteration - Avg: -4.6 - Max: -3.0
* 100kth Iteration - Avg: -4.6 - Max: -3.0

* 120kth Iteration - Avg: -4.27 - Max: -1.0
* 140kth Iteration - Avg: -3.27 - Max: -1.0
* 160kth Iteration - Avg: -2.27 - Max: 4.0
* 180kth Iteration - Avg: -2.27 - Max: 4.0
* 200kth Iteration - Avg: -1.73 - Max: 5.0

* 220kth Iteration - Avg: 1.93 - Max: 10.0
* 240kth Iteration - Avg: 3.87 - Max: 21.0
* 260kth Iteration - Avg: 6.47 - Max: 32.0
* 280kth Iteration - Avg: 21.33 - Max: 50.0
* 300kth Iteration - Avg: 42.93 - Max: 123.0

* 320kth Iteration - Avg: 162.67 - Max: 734.0
* 340kth Iteration - Avg: 126.07 - Max: 310.0
* 360kth Iteration - Avg: 62.07 - Max: 175.0
* 380kth Iteration - Avg: 23.2 - Max: 69.0
* 400kth Iteration - Avg: 43.87 - Max: 259.0

* 420kth Iteration - Avg: 37.0 - Max: 147.0
* 440kth Iteration - Avg: 159.87 - Max: 469.0
* 460kth Iteration - Avg: 268.47 - Max: 1459.0
* 480kth Iteration - Avg: 72.53 - Max: 170.0
* 500kth Iteration - Avg: 66.73 - Max: 214.0

On average, the bird reaches more than a score of 110 on 100 games.
