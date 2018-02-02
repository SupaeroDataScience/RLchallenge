# You're not allowed to change this file
from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np
import pygame
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import resize
from time import time
import pickle
import random

filehander = open("images.pickle", "rb")
images = pickle.load(filehander)
filehander.close()

for i in range(10):
    # print(np.shape(random.choice(images)))
    plt.imshow(random.choice(images),  cmap="gray")
    plt.show()
