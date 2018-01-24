import numpy as np
import sys
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import resize

from CNN import CNN

CNN_GLOBAL = None # Contains a model in order to choose an action thanks to pixels

class Policy3():
    """ Class Policy3
    Is used for choosing an action thanks to pixels """


    def __init__(self, state, screen):
        self.action = None
        self.state = state
        self.screen = screen
        self.filepathCNN = "model_dql_flappy3_dense.dqf"
        self.model = self.init_CNN()

    def init_CNN(self):
        ''' Load a CNN into a global variable
        If this variable exists, do nothing except returning the global variable '''
        global CNN_GLOBAL
        if not CNN_GLOBAL: # have to load it once
            cnn = CNN()
            try:
                cnn.load(self.filepathCNN)
                CNN_GLOBAL = cnn.model # Add it to a global_variable

            except IOError:
                print(' !! File not found : ', self.filepathCNN)
                sys.exit()

        return CNN_GLOBAL


    def transform_screen(self):
        screen_cut = self.screen[50:-1, 0:400] # cut
        screen_grey = 256 * (rgb2gray(screen_cut)) # in gray
        output = resize(screen_grey, (84, 84)) # resize
        output = np.stack([output, output, output, output], axis=-1)
        return output


    def plot_screen(self, screen):
        plt.imshow(screen)
        plt.show()


    def get_action(self, screen_modified):
        ''' Use the CNN_GLOBAL with screen_modified
        Should return an action to do '''
        neural_value = self.model.predict(screen_modified)
        if round(neural_value) == 1 :
            return 119
        else :
            return None
