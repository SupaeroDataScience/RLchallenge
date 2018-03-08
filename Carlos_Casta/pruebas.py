# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 12:17:01 2018

@author: Carlos CASTA
"""

import numpy as np

materias = {}
materias["lunes"] = [550, 220]
materias["martes"] = [6201]
materias["miércoles"] = [6103, 7540]
materias["jueves"] = []
materias["viernes"] = [6201]

print (max(materias ["lunes"]))


actions = [0,119,855]

print (np.argmax(actions))

matrizQ = dict()

matrizQ["statep"] = [0 , 5]

maxQ = np.argmax(matrizQ["statep"])
print (maxQ)