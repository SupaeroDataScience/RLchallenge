# Carlos CASTA - Flappy Bird

This projet was developed between February and March 2018 during the course Machine Learning at ISAE-SUPAERO as a final challenge for the engineering students.

A simple description of the projet would be: A Flappy Bird bot in Python, that learns from each game played via Q-Learning.


# Installation Dependencies:

Python 3.6

Pygame

PLE (PyGame Learning Environment)

Note that this version of FlappyBird in PLE has been slightly changed to make the challenge a bit easier: the background is turned to plain black, the bird and pipe colors are constant (red and green respectively).

#Summary of the projet


The player, Flappy Bird in this case, performs a certain action in a state. It then finds itself in a new state and gets a reward based on that. There are many variants to be used in different situations: Policy Iteration, Value Iteration, Q Learning, etc.

With every game played, we observe the states the bird has been in and the actions it took. With regards to their outcomes, we give a punishment or a reward to the state-action pairs. After playing the game numerous times (5000 more or less), the bird is able to obtain high scores.

#
