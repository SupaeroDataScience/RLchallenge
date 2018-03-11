#### Flappy Bird policy selection function
#### Convolutional neural network

#%%
# Imports
import numpy as np
from keras.models import load_model

import skimage as skimage
from skimage import color, transform, exposure

ACTIONS = [0, 119] # valid actions
          
model = load_model("model.h5")
np.seterr(divide='ignore', invalid='ignore')

#%%
# Function to select the optimal policy.
def FlappyPolicy(state, screen):
    
    x_t = skimage.color.rgb2gray(screen)
    x_t = skimage.transform.resize(x_t,(80,80))
    x_t = skimage.exposure.rescale_intensity(x_t,out_range=(0,255))

    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    #In Keras, need to reshape
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  #1*80*80*4

                 
    q = model.predict(s_t)       #input a stack of 4 images, get the prediction
    max_Q = np.argmax(q)
    action_index = max_Q
    a_t = ACTIONS[action_index]
    
    # We return the selected action
    return a_t