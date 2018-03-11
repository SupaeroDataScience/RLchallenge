class constantes : # Fix main constants
   
    # memory buffer constants
    replay_memory_size = 200000 # number of previous transitions to remember
    mini_batch_size = 32
   
    # Learning constants
    gamma = 0.99
    total_steps = 200000 # The best network was obtained after 65000 steps
    observation = 5000.
    explore = 1000000. # frames over which to anneal epsilon
    final_eps = 0.001 # final value of epsilon
    initial_eps = 0.1 # starting value of epsilon
   
    # Optimizer's constants
    alpha = 1e-4 # learning rate
    beta_1 = 0.9
    beta_2 = 0.999
   
    # Evaluation constants
    evaluation_period = 5000 # Ealuation of the deep q network every 5000 steps
    nb_epochs = total_steps // evaluation_period
    epoch=-1

    
