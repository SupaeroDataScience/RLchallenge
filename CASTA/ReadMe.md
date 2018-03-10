# Carlos CASTA - Flappy Bird

This projet was developed between February and March 2018 during the F-SD311 Algorithms in Machine Learning course at ISAE-SUPAERO as a final challenge for the engineering students.

A simple description of the projet would be: A Flappy Bird bot in Python, that learns from each game played via Q-Learning.

# Summary of the projet

With every game played, we observe the states that the bird has been in and the actions it took. With regards to their outcomes, we give a punishment or a reward to the state-action pairs. After playing the game numerous times (5000 more or less), the bird is able to obtain high scores.

# Reinforcement Learning

The intuitive thought about reinforcment learning is to consider situation in which we (or an agen, Flappy Bird in this case) interact with the environement via a sequence of observations, actions and rewards.

The goal of reinforcement learning is to maximize the total pay-off (reward). In Q-learning, which is off-policy, we use the bellman equation as an iterative update

U(s)=R(s)+γ∑P(st+1|s,a)maxa∈A(s)U(st+1)
U(s)=R(s)+γ∑P(st+1|s,a)maxa∈A(s)U(st+1)


Where $U(s)$, $R(s)$ are the *utility* and the *reward* of the *state* $s$, $\gamma$ the *discount factor* and $P(s_{t+1} | s, a)$ the conditional probability to be in state $s_{t+1}$ if I do the action $a$ in the state $s$.

I used Q Learning because it is a model free form of reinformcent learning. That means that I didn't have to model the dynamics of Flappy Bird; how it rises and falls, reacts to clicks and other things of that nature.

# Installation Dependencies:

Python 3.6

Pygame

PLE (PyGame Learning Environment)

Note that this version of FlappyBird in PLE has been slightly changed to make the challenge a bit easier: the background is turned to plain black, the bird and pipe colors are constant (red and green respectively).
