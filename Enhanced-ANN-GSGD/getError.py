import numpy as np
import pandas as pd
import math
import os
from math import exp
from propagation import forward_propagate, backward_propagate_error
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier

def custom_train(network, xinst, yinst, n_outputs):
    the_unactivateds, the_activateds = forward_propagate(network, xinst)
    outputs = the_activateds[-1]
    expected = [0 for i in range(n_outputs)]
    expected[int(yinst)] = 1
    #get mse 
    errorstotal = 0.00
    for accurateval, actual in zip(expected, outputs):
        diffval = actual - accurateval
        diffval = diffval **2
        errorstotal += diffval
    return errorstotal/n_outputs
 
def getError(idx, x, y, network, n_outputs):
    mean_squared_error = custom_train(network,x[idx], y[idx],n_outputs) 
    return mean_squared_error

def getErrorCrossEntropy(idx, x, y, network, n_outputs):
    the_unactivateds, the_activateds = forward_propagate(network, x[idx])
    y_pred = the_activateds[-1]
    expected = [0 for i in range(n_outputs)]
    expected[int(y[idx])] = 1
    y_true = expected
    error = log_loss(y_true, y_pred)
    return error