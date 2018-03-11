# RL challenge

Several reinforcement algorithms and extensions where explored while conducting this assignment. Most of them achieved above human performance scores (well above the 15 score demanded by the original assignment), while one didn't converge. The source code, containing the training scripts, reflects all the exploration that was done. The explored algorithms and their status are:

1. Lambda SARSA - Status: OK
   * on engineered state
2. Q-learning - Status: OK
   * pn engineered state
3. Q-learning with NN - Status: NOT OK
   * on full state
   * with Prioritized Experience replay \[ ICLR 2016 \]
   * and Double QN \[Deep Reinforcement Learning with Double Q-learning AAAI 2016\]
4. DQN - Status: OK
   * on pixels

## To run

In order to evaluate the performance of the trained algorithms, you should open the corresponding folder:
* `0_Lambda SARSA/`
* `1_Q learning/`
* `2_DQN/`

and run `run.py`. Note that only the algorithms that converged successfully are available to evaluate.


## Source code

All the source code is contained in the folder `training/`
