import numpy as np
import sys
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import load_model

CNN_GLOBAL = None # Contains a model in order to choose an action thanks to pixels

class Policy3():
    """ Class Policy3
    Is used for choosing an action thanks to pixels """

    def __init__(self, state, screen, path_model):
        self.action = None
        self.state = state
        self.screen = screen
        self.filepathCNN = path_model
        self.model = self.init_CNN()

    def init_CNN(self):
        ''' Load a CNN into a global variable
        If this variable exists, do nothing except returning the global variable '''
        global CNN_GLOBAL
        if not CNN_GLOBAL: # have to load it once
            try:
                CNN_GLOBAL = load_model(self.filepathCNN)

            except IOError:
                print(' !! File not found : ', self.filepathCNN)
                sys.exit()

        return CNN_GLOBAL

    def transform_screen(self):
        screen_cut = self.screen[50:-1, 0:400] # cut
        screen_grey = 256 * (rgb2gray(screen_cut)) # in gray
        output = resize(screen_grey, (84, 84), mode='constant') # resize
        output = np.stack([output, output, output, output], axis=-1)
        return output

    def plot_screen(self, screen):
        plt.imshow(screen)
        plt.show()

    def get_action(self, screen_modified):
        '''
        Use the CNN_GLOBAL with screen_modified
        Should return an action to do
        '''
        neural_value = self.model.predict(np.array([screen_modified]))
        print(neural_value)
        return np.argmax(neural_value)

