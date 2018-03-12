import numpy as np
from keras.models import Sequential, load_model
model = load_model("best_model.dqf")
def FlappyPolicy(state, screen):
    q = model.predict(np.array(list(state.values())).reshape(1,len(state)))
#         q = self.model.predict(screen.reshape(1, screen.shape[0], screen.shape[1], screen.shape[2]))
#    print(q)
        
    return(np.argmax(q)*119)
#     return np.random.randint(0,1)*119


