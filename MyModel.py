# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 14:03:17 2022

@author: xredin00
"""
import numpy as np

def MyModel(data):

#     Funkce slouzi k implementaci nauceneho modelu. Vas model bude ulozen v samostatne promenne a se spustenim se aplikuje
#     na vstupni data. Tedy, model se nebude pri kazdem spousteni znovu ucit. Ostatni kod, kterym doslo k nauceni modelu,
#     take odevzdejte v ramci projektu.

#Vstup:             data:           vstupni data reprezentujici 1
#                                   objekt (1 pacienta, 1 obrazek, apod.). 

#Vystup:            output:         zarazeni objektu do tridy

    output = np.random.randint(0,high=10)
    
    return output