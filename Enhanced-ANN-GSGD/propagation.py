import numpy as np
import pandas as pd
import math
import os
from math import exp
from random import seed
from random import randrange
from random import random  # check if seed is used or not

def forward_propagate(network, row):
    the_unactivateds = []
    the_activateds = []
    for layer in network:
        row = np.append(row, np.ones((1,),dtype=np.float64),axis=0)  #adding bias to input
        non_activated = row.T @ layer  # i think this needs to be updated to exclude bias or something sortof
        row = transfer(non_activated)
        the_unactivateds.append(non_activated.copy())
        the_activateds.append(row.copy())
    
    return the_unactivateds, the_activateds
        
# Transfer neuron activation
def transfer(non_activated):
    activated = []
    for i in range(len(non_activated)):
        input_activated = 1.0 / (1.0 + exp(-i))
        activated.append(input_activated)
        
    return np.array(activated)

def backward_propagate_error(network, expected, the_activateds):
    the_deltas = list()
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            #error = network[i+1] @ the_deltas[len(the_deltas)-1]    ## only claculate the error for the weights that is coming from successer layer neuron to the neuron in this preceding layer       update this
            errors =  network[i+1][:-1] @ the_deltas[len(the_deltas) -1]   # network[i+1][:-1] is so that we do not get bias weight which will not be used here (code is removing bias weiths which is at very end)
            print('is not final layer')
        else:
            errors = np.subtract(the_activateds[i], expected) # this error is partial error complete error of neuron is got when multiply this with the derivative of the transfer function
        
        get_transfer_derivative = transfer_derivative(the_activateds[i])
        this_delta = np.multiply(errors, get_transfer_derivative.copy())
        the_deltas.append(this_delta.copy()) 
    return list(reversed(the_deltas))
        
def transfer_derivative(the_activateds_layer):
    ones = np.ones((the_activateds_layer.size,), dtype=np.float64)
    subtracted_value =  np.subtract(ones, the_activateds_layer)
    resp = np.multiply(the_activateds_layer,subtracted_value)
    return resp

def update_weights(network, row, l_rate, deltas, the_activateds):
    for i in range(len(network)):
        layer_input = np.append(row, np.ones((1,),dtype=np.float64),axis=0)  #adding bias to input
        if i > 0:
            layer_input = np.append(the_activateds[i-1], np.ones((1,),dtype=np.float64),axis=0)  #adding bias to input
            
        rowscount = layer_input.shape
        #network[i] = np.subtract(network[i], np.multiply(np.array([l_rate for r in range(rowscount[0])]), np.multiply(layer_input.reshape(len(layer_input),1), deltas[i].reshape(1,len(deltas[i])))))  #layer_weights =  layer_weights - l_rate * row * deltas[i]
        
        old_weights_matrix = network[i]
        l_rate_matrix =  np.array([l_rate for r in range(rowscount[0])]).reshape(rowscount[0],1)
        layer_input_matrix = layer_input.reshape(len(layer_input),1)
        delta_matrix = deltas[i].reshape(1,len(deltas[i]))
        
        gradient_essent = np.multiply(np.multiply(l_rate_matrix, layer_input_matrix), delta_matrix)
        network[i] = np.subtract(old_weights_matrix, gradient_essent)
        
        #print(network[i])