import numpy as np
import pandas as pd
import math
from propagation import forward_propagate

from getError import getError


def validate(inputVal, network, givenOut, nfc, pocket, n_outputs):
    xval = inputVal
    doTerminate = False
    
    #get predicted value
    predicted = get_predictions(network, xval)
    actual = givenOut
    totCorrect = accuracy_metric(actual, predicted)
    
    SR = totCorrect/len(xval[:,1])  

    if SR > 0.5:
        if len(pocket.weights) == 0:  # try:
            # print('--->')
            # print(nfc)
            pocket.weights = network
            pocket.sr = SR
            pocket.nfc = nfc

        # except IndexError:
        else:
            if pocket.sr < SR:
                # print(W)
                pocket.weights = network
                pocket.sr = SR
                pocket.nfc = nfc

        # print(SR)
        if SR > 110.85:  # percentage defined after decimal. <85% would be 0.85>
            doTerminate = True
        else:
            False

    N = np.size(inputVal, axis=0)  # number of cols    
    E = 0
    for IN in range(N):
        nErr = getError(IN, inputVal, givenOut, network, n_outputs)
        E = E + nErr

    E = E/N

    return doTerminate, SR, E, pocket  # PocketGoodWeights, doTerminate, SR, E, pocket


def validateSGD(inputVal, network, givenOut, n_outputs):
    xval = inputVal
    doTerminate = False
    
    #get predicted value
    predicted = get_predictions(network, xval)
    actual = givenOut
    totCorrect = accuracy_metric(actual, predicted)
        
    SR = totCorrect/len(xval[:,1])

    N = np.size(inputVal, axis=0)  # number of cols
    E = 0
    for IN in range(N):
        nErr = getError(IN, inputVal, givenOut, network, n_outputs)
        E = E + nErr

    E = E/N

    return doTerminate, SR, E


# Make a prediction with a network
def predict(network, row):
	unactivated_outputs, activated_output = forward_propagate(network, row)
	return np.argmax(activated_output[-1])

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i][0] == predicted[i][0]:
			correct += 1
	return correct  


def get_predictions(network, xts):
    predictions_test = list()
    for row in xts:
        prediction = predict(network, row)
        predictions_test.append([prediction])
    return (predictions_test)