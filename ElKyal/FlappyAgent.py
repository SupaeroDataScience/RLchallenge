import numpy as np
from keras.models import Sequential, load_model

model = load_model("model.dqf")

def FlappyPolicy(state, screen):
    q = model.predict(np.array(list(state.values())).reshape(1,len(state)))

        
    return(np.argmax(q)*119)
