# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 23:04:31 2018

@author: Arnaud
"""
import pickle
f_myfile = open('Q_function_5000_ite.pickle', 'rb')
Qql = pickle.load(f_myfile)  
f_myfile.close()

