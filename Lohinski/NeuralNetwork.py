from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization

import numpy as np


class NeuralNetwork:
    def __init__(self, batch_size=40, epochs=1, verbose=False):
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        model = Sequential()

        model.add(Dense(8, kernel_initializer='lecun_uniform', input_shape=(8,)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=1))
        model.add(Dense(64, kernel_initializer='lecun_uniform'))
        model.add(Activation('relu'))
        model.add(Dropout(rate=0.2))
        model.add(Dense(32, kernel_initializer='lecun_uniform'))
        model.add(Activation('relu'))
        model.add(Dropout(rate=0.2))
        model.add(Dense(16, kernel_initializer='lecun_uniform'))
        model.add(Activation('relu'))
        model.add(Dropout(rate=0.2))
        model.add(Dense(2, kernel_initializer='lecun_uniform'))
        model.add(Activation('linear'))
        model.compile(loss='mean_squared_error', optimizer="rmsprop")

        self.model = model

    def predict(self, data):
        x = np.array(list(data.values())).reshape(1, len(data))
        return self.model.predict(
            x=x, batch_size=self.batch_size, verbose=self.verbose
        )

    def train(self, data, output):
        if isinstance(data, np.ndarray):
            x = data
        else:
            x = np.array(list(data.values())).reshape(1, len(data))
        return self.model.fit(
            x=x, y=output,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=self.verbose
        )
