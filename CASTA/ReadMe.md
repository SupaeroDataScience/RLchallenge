# Carlos CASTA - Flappy Bird

This project was developed between February and March 2018 during the F-SD311 Algorithms in Machine Learning course at ISAE-SUPAERO as a final challenge for the engineering students.

A simple description of the projet would be: A Flappy Bird bot in Python, that learns from each game played via Q-Learning.

# Summary of the projet

With every game played, we observe the states that the bird has been in and the actions it took. With regards to their outcomes, we give a punishment or a reward to the state-action pairs. After playing the game numerous times (5000 more or less), the bird is able to obtain high scores.

# Reinforcement Learning

The intuitive thought about reinforcment learning is to consider situation in which we (or an agen, Flappy Bird in this case) interact with the environement via a sequence of observations, actions and rewards.

The goal of reinforcement learning is to maximize the total pay-off (reward). In Q-learning, which is off-policy, we use the bellman equation as an iterative update.

Instead of learning an utility associate to observe states, the agent could learn a relation action - value.  Q(s,a),  nammed also Q-Value, represents the value of an action  [a]  in a state  [s].

Thus, combining the bellman equation and temporal difference learning approach we manage to finally have our Q-learning equation:

Q[s,a] ←Q[s,a] + α(reward + γ * max' Q[s',a'] - Q[s,a])

The Q-Value is updated at each step where an action  [a]  is done in the state  [s] . Q-Values are very wrong in early stages of learning. Nevertheless, the estimations get more and more accurate at every iterations and it has beeen shown that the Q function converges to real Q-Values.

# State Space

I discretized my space over three parameters.

- The vertical distance from lower pipe to the bird

- The horizontal distance from next pair of pipes to the bird

- The bird's speed

# Actions

For each state, there are two possible actions:

- Click [119]

- Do Nothing [0]

# Reward

- +1 if Flappy Bird passes one pairs of pipes

- -5 if Flappy Bird deads

# Installation Dependencies:

- Python 3.6

- Pygame

- PLE (PyGame Learning Environment)

Note that this version of FlappyBird in PLE has been slightly changed to make the challenge a bit easier: the background is turned to plain black, the bird and pipe colors are constant (red and green respectively).
