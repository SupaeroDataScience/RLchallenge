# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 15:45:02 2018

@author: Nico
"""
from FlappyAgent import FlappyQLearner
import matplotlib.pyplot as plt

agent = FlappyQLearner(250)
Q, loss, scores, count = agent.learn(True,"5000_epochs_multistep_initialized")
agent.save("4200_epochs_multistep_initialized_FINAL")
plt.plot(scores)