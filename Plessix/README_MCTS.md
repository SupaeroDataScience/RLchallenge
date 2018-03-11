# The Challenge :

The goal is to learn to play Flappy Bird with reinforcement learning : the agent must successfully navigate through gaps between pipes.
The solution proposed here is using Monte Carlo tree serach (MCTS). 

# The Approach :

The concept is to create a search tree node by node, according to the outcomes of the simulated payouts.

In the tree:
- **Nodes** represent game states
- **Edges** represent possible actions from a state game to an other one

Each node stores two values :
- The **Play Value** : how many times the node has been played
- The **Win Value** : quantitative payoff associated to the node (the reward process will be described later)

# Reward Process

We chose to adopt the following strategy regarding the update of the Win Value :
- a reward of 1 is given to all the states leading to pass a pipe
- a penalty is given to a certain number of states leading to a collision (ie a terminal state)

# Results

The best results have been obtained with the following parameters :
  - a penalty of **-5.0** is applied to the 9 last states leading to a terminal state
  - the training is performed on 35 000 games
 The trained tree associated is strored in the file "trained_tree"
 
 The tree contains then approximately 42 000 nodes (on the 89 000 possible states with the state reduction we performed)

# State Reduction

Initially, the function getGameState() implemented in the library returns a 8-state state vector. 
To simplify the problem, we will perform some feature-engineering:
<ol> 
<li> we will take only 3 elements : 
    <ul>
    <li> player_vel
    <li> next_pipe_dist_to_player
    <li> height (distance between next_pipe_top_y and player_y) 
    </ul>
<li> these states are discretised and scaled
</ol>

# Asset of the MCTS

The main asset is that there is only need to store the data of the reachable states.
Moreover, the construction strategy will discourage the discovering of branches leading to terminal states.

# Discussion

Afterwards, some tests have been realized to evaluate the impact of the decisions we made on the parameters choices:

<ul>
<li>**Adding one state** :  
We tried to add the next-next pipe distance to the player.
The goal is to be able to anticipate the location of the next pipe, and chose the action accordingly.
With the scaling, the number of possible states is then multiplied by 71, which increase the number trainings needed.
At 50 000 games, the average score was still 6.0.

</ul>
