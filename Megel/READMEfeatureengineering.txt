Solution found for FlappyBird feature engineering: multi-step Q learning

In class "FlappyQLearner", some Q learning was performed on handcrafted features.
The discretized state is described in function "FlappyQLearner.convert_state"

Training was performed in 2 successives steps with no noise introduced:

-> 4000 epochs with a one-step return Q-learning performed at the end of each game,
with each of the 9 last state-action pairs penalized with - 1000 as reward.
At this stage, the agent can score 900 as well as 0 because it still cannot deal with the notion of speed
when being confronted to high altitude differences. However, it has "understood" the rules of that game and identifies quickly how to lose.
Convergence to optimal policy is low and uncomplete

-> 200 more epochs with a multi-step Q learning performed within a game on each state-action visited taking into account the entire game.
Defeat reward is lowered to -50 at moment of defeat. This setting enables the agent to understand that a given succession of actions led to defeat
During this training, the agent is more stable and has average score of 250 with low variance. It handles its speed much more efficiently.
It is still trapped by some unseen combinations, my suggestion is that exploration of useful state is not totally complete and 
that a random choice phase could be added.

The idea was to provide fair scores with only 4200 game training with a method that presents both simple state space and efficient off-policy Q-updates with no need for a GLIE actor