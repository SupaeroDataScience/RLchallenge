# Implementations
## Q-learning approach with temporal differences TD(0)
This algorithm approximates the Q matrix associated to the optimal policy by choosing actions greedily and updating it using the best next Q value independently of the taken policy (off-policy).

## SARSA approach with temporal differences TD($\lambda$)
The SARSA algorithm inferes the value of the problem's optimal policy's Q matrix by chosing actions greedily and updating the value of the Q on the evaluated policy.

This allows for using the TD($\lambda$) value estimator, which allows for a much faster propagation of the information and thus a faster convergence.

## Q-learning using a neural network as an approximating function
It inferes the Q matrix values through a neural network. This implementation uses a memory replay.

**This case does not work at the moment** due to a lack of an appropriate hyperparameter tunning. Once properly working it will be naturably extensible to use all the PLE's state variables.

# Acknowledgements
The theoretical foundations for this work are based on Emmanuel Rachelson's course on machine learning

Some implementation details and hyperparameters based on the work of:
https://github.com/chncyhn/flappybird-qlearning-bot