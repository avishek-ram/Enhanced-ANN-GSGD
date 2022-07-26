import numpy as np
import pandas as pd
import math
import os
import copy
from math import exp

def forward_propagate(network, row):
    the_unactivateds = []
    the_activateds = []
    for layer in network:
        row = np.append(row, np.ones((1,),dtype=np.float64),axis=0)  #adding bias to input
        non_activated = row.T @ layer
        row = transfer(non_activated)
        the_unactivateds.append(non_activated.copy())
        the_activateds.append(row.copy())
    
    return the_unactivateds, the_activateds
        
# Transfer neuron activation
def transfer(non_activated):
    # activated = []
    # for i in non_activated:
    #     input_activated = 1.0 / (1.0 + exp(-i))
    #     activated.append(input_activated)
    # return np.array(activated)
    numerator = np.ones(shape=non_activated.size)
    denominator = np.add(np.ones(shape=non_activated.size), np.exp(-1 * non_activated))
    activation_matrix = np.divide(numerator, denominator)  #sigmoid function
    return activation_matrix

def backward_propagate_error(network, expected, the_activateds):
    the_deltas = list()
    for i in reversed(range(len(network))):
        errors = list()
        if i != len(network)-1:
            errors =  network[i+1][:-1] @ the_deltas[len(the_deltas) -1]   # network[i+1][:-1] is so that we do not get bias weight which will not be used here (code is removing bias weights which is at very end)
        else:
            errors = np.subtract(the_activateds[i], expected)    #this is the derivative of the Mean Squared Error function f'(MSE) =  actual - predicted ; MSE = 1/2 * sum(actual - predicted)^2
        
        get_transfer_derivative = transfer_derivative(the_activateds[i])
        this_delta = np.multiply(errors, get_transfer_derivative.copy())
        the_deltas.append(this_delta.copy()) 
    return list(reversed(the_deltas))
        
def transfer_derivative(the_activateds_layer):
    ones = np.ones((the_activateds_layer.size,), dtype=np.float64)
    subtracted_value =  np.subtract(ones, the_activateds_layer)
    resp = np.multiply(the_activateds_layer,subtracted_value)
    return resp

def update_weights(network, row, l_rate, deltas, the_activateds, lamda, d):
    for i in range(len(network)):
        layer_input = np.append(row, np.ones((1,),dtype=np.float64),axis=0)  #adding bias to input
        if i > 0:
            layer_input = np.append(the_activateds[i-1], np.ones((1,),dtype=np.float64),axis=0)  #adding bias to input

        old_weights_matrix = copy.deepcopy(network[i])
        l_rate_matrix =  np.array([l_rate for r in range(old_weights_matrix.size)]).reshape(old_weights_matrix.shape)
        layer_input_matrix = layer_input.reshape(len(layer_input),1)
        delta_matrix = deltas[i].reshape(1,len(deltas[i]))
                
        #formula-> weights = w' = old_weights -  lr * xi * deltaj
        step1 = layer_input_matrix @ delta_matrix  # xi * delta;
        step2 = np.multiply(step1,l_rate_matrix)  #lr * xi * deltaj
        default_weights = np.subtract(old_weights_matrix, step2)
        #network[i] = np.subtract(old_weights_matrix, step2)  # new weights nonregularized
        network[i] = np.add(default_weights,((np.float64(lamda)/np.float64(d)) * old_weights_matrix))   #new Weights- Regularized using L2
        