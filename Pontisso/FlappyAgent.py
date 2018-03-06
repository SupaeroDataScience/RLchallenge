import numpy as np

def FlappyPolicy(state, screen):
  
    batchSize = 2

    # load model
    
    qval = model.predict(np.array(list(state.values())).reshape(1,8), batch_size=batchSize) 
    qval_av_action = qval[0]
    action = (np.argmax(qval_av_action))*119
    #print("qval = ", qval, "state =" , np.array(list(state.values())).reshape(1,8))
    return action



