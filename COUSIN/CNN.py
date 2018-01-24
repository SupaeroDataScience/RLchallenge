from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten


class CNN():
    ''' Class gathering aln information needed to run a CNN
    Functions:  Save, Load, and __init__
    '''

    def __init__(self):
        self.model = None

    def init(self):
        model = Sequential()
        # 1st layer
        model.add(Conv2D(filters=16, kernel_size=(8, 8), strides=4, activation="relu", input_shape=(84, 84, 4)))
        # 2nd layer
        model.add(Conv2D(filters=32, kernel_size=(4, 4), strides=2, activation="relu"))
        model.add(Flatten())
        # 3rd layer
        model.add(Dense(units=256, activation="relu"))
        # output layer
        model.add(Dense(units=4, activation="linear"))

        model.compile(optimizer="rmsprop", loss="mean_squared_error")
        self.model = model

    def load(self, filepath):
        ''' Function load used to instantiate a global variable
        This variable contains the model to play
        Only used for playing. Or to load for training again...
        '''

        self.model = load_model(filepath)
        print(self.model)

    def save(self, filepath):
        self.model.save(filepath)
