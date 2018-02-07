from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
import pickle
import numpy as np
import sys

class CNN():
    ''' Class gathering aln information needed to run a CNN
    Functions:  Save, Load, and __init__
    '''

    def __init__(self, directory, file_model, file_buffer, file_step, file_score):
        self.model = None
        self.buffer = None
        self.step = None
        self.score = None
        self.path_model = directory+file_model
        self.path_buffer = directory + file_buffer
        self.path_step = directory+file_step
        self.path_score = directory+file_score


    def init(self):
        print("##########################################")
        print("Initialization of a new CNN...")
        self.init_CNN()
        self.step = 0
        print("CNN initialized!")
        print("##########################################\n")


    def init_CNN(self):
        model = Sequential()
        # 1st layer
        model.add(Conv2D(filters=16, kernel_size=(8, 8), strides=4, activation="relu", input_shape=(84, 84, 4)))
        # 2nd layer
        model.add(Conv2D(filters=32, kernel_size=(4, 4), strides=2, activation="relu"))
        model.add(Flatten())
        # 3rd layer
        model.add(Dense(units=256, activation="relu"))
        # 4th layer
        model.add(Dense(units=8, activation="relu"))
        # output layer
        model.add(Dense(units=1, activation="linear"))
        model.compile(optimizer="rmsprop", loss="mean_squared_error")
        self.model = model


    def load(self):
        ''' Function load used to instantiate a global variable
        This variable contains the model to play
        Only used for playing. Or to load for training again...
        '''
        print("##########################################")
        print("Loading CNN, Buffer, current Step and Score...")
        try:
            self.model = load_model(self.path_model)
            print("CNN loaded!")
        except IOError:
            print("{} not found!".format(self.path_model))
            sys.exit()

        try:
            self.step = np.load(self.path_step)
            print("Step loaded!")
        except IOError:
            print("{} not found!".format(self.path_step))
            sys.exit()

        try:
            with open(self.path_buffer, 'rb') as fid:
                self.buffer = pickle.load(fid)
            print("Buffer loaded!")
        except IOError:
            print("{} not found!".format(self.path_buffer))
            sys.exit()

        try:
            self.score = np.load(self.path_score)
            print("Score loaded!")
        except IOError:
            print("{} not found!".format(self.path_score))
            sys.exit()

        print("==> FILES CORRECTLY LOADED!")
        print("##########################################\n")

    def save(self):
        '''

        :return:
        '''
        print("##########################################")
        print("Saving CNN, Buffer, current Step and Score...")
        self.model.save(self.path_model)
        print("CNN saved!")
        with open(self.path_buffer, 'wb') as fid:
            pickle.dump(self.buffer, fid, pickle.HIGHEST_PROTOCOL)
            print("Buffer saved!")
        np.save(self.path_step, self.step)
        print("Step saved!")
        np.save(self.path_score, self.score)
        print("Score saved!")
        print("==> FILES CORRECTLY SAVED!")
        print("##########################################\n")
