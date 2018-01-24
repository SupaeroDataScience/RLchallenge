import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten


class CNN():
    ''' Class gathering aln information needed to run a CNN
    Functions:  Save, Load, and __init__
    '''

    def __init__(self):
        self.dqn = self.init()

    def init(self):
        dqn = Sequential()
        # 1st layer
        dqn.add(Conv2D(filters=16, kernel_size=(8, 8), strides=4, activation="relu", input_shape=(84, 84, 4)))
        # 2nd layer
        dqn.add(Conv2D(filters=32, kernel_size=(4, 4), strides=2, activation="relu"))
        dqn.add(Flatten())
        # 3rd layer
        dqn.add(Dense(units=256, activation="relu"))
        # output layer
        dqn.add(Dense(units=4, activation="linear"))

        dqn.compile(optimizer="rmsprop", loss="mean_squared_error")
        return dqn

    def load(self, file):
        pass

    def save(self, file):
        pass


class Policy():

    def __init__(self, state, screen):
        self.action = None
        self.state = state
        self.screen = screen
        self.dqn = CNN()


    def transform_screen(self):
        print("Initial screen : ", self.screen.shape)
        output = 256 * (rgb2gray(self.screen))[:, 0:400] # in gray + cut
        output = resize(output, (84, 84)) # resize
        print("Resize : ", output.shape)
        output = np.stack([output, output, output, output], axis=-1)
        print("Stack : ", output.shape)
        return output


    def plot_screen(self):
        plt.imshow(self.screen)
        plt.show()


    def get_action(self):
        return None