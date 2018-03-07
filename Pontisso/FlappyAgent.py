from keras.models import load_model
import numpy as np

model = load_model("bestmodel.dqf")

batchSize = 2

#This function returns the best action to perform given the state of the bird
def FlappyPolicy(state, screen):
    
    qval = model.predict(np.array(list(state.values())).reshape(1,8), batch_size=batchSize) 
    qval_av_action = qval[0]
    action = (np.argmax(qval_av_action))*119
    return action




