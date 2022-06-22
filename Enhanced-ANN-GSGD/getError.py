import numpy as np
import pandas as pd
import math
import os
from math import exp
from propagation import forward_propagate, backward_propagate_error

#custom code
def custom_train(network, xinst, yinst, n_outputs): #will get delta/  neuron error
    the_unactivateds, the_activateds = forward_propagate(network, xinst)
    expected = [0 for i in range(n_outputs)]
    expected[yinst[0]] = 1
    the_deltas =backward_propagate_error(network, np.array(expected), the_activateds)  # all neuron errors
    return the_deltas

def getError(idx, x, y, network, n_outputs):
    errors = custom_train(network,x[idx, :], y[idx, :],n_outputs) #will give all layer neuron errors
    output_neurons_error = errors[len(errors)-1]
    mean_squared_error = np.sum([neuron**2 for neuron in output_neurons_error]) / len(output_neurons_error)
    return mean_squared_error