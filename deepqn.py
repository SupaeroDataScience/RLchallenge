#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 16:56:33 2018

@author: paul
"""

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten



def create_model(plot=False,name='') : 
  dqn = Sequential()
  #1st layer
  dqn.add(Conv2D(filters=16, kernel_size=(8,8), strides=4, activation="relu", input_shape=(122,205,4)))
  #2nd layer
  dqn.add(Conv2D(filters=32, kernel_size=(4,4), strides=2, activation="relu"))
  dqn.add(Flatten())
  #3rd layer
  dqn.add(Dense(units=256, activation="relu"))
  #output layer
  dqn.add(Dense(units=2, activation="linear"))
  
  dqn.compile(optimizer="rmsprop", loss="mean_squared_error")
  
  if plot :
    plot_model(dqn, to_file=name+'.png', show_shapes=True)
    
  return dqn 

model = create_model()