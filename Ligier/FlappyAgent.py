#import matplotlib.pyplot as plt
from skimage import transform,color
import numpy as np
#from keras.models import Sequential
#from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.models import load_model

'''deepQnet = Sequential()
deepQnet.add(Conv2D(filters=16, kernel_size=(8,8), strides=4,
                        activation="relu", input_shape=(72,101,4)))
deepQnet.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid',
                 data_format=None))
deepQnet.add(Conv2D(filters=32, kernel_size=(4,4), strides=2,
                        activation="relu"))
deepQnet.add(Flatten())
deepQnet.add(Dense(units=256, activation="relu"))
deepQnet.add(Dense(units=2, activation="linear"))
deepQnet.compile(optimizer='adam', loss='mean_squared_error')'''
deepQnet = load_model('model.h5')

def FlappyPolicy(state, screen):
    global deepQnet
    # Crop at 404 px and resize by dividing by 4
    screen = transform.resize(color.rgb2gray(screen[:,:404,:]),(screen.shape[0]/4,101))
    frameStack = np.stack([screen]*4,axis=-1)

    a = deepQnet.predict(np.expand_dims(frameStack,axis=0))

    #plt.imshow(screen, cmap='gray')
    #plt.show()

    return None # Should return an action
