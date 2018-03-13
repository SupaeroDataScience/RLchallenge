
import numpy as np
from keras.models import load_model

model = load_model("model_flappy_best_score.dqf")

def FlappyPolicy(state, screen):
    qval = model.predict(np.array(list(state.values())).reshape(1,8), batch_size=1) 

    return 119 * (np.argmax(qval[0]))