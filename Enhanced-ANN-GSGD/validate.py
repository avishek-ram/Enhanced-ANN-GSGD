import numpy as np
import pandas as pd
import torch
import torchmetrics
import copy
from getError import *


def validate(inputVal, network, givenOut, nfc, pocket, n_outputs, epoch, loss_function):
    xval = inputVal
    doTerminate = False
    
    #get predicted value
    predicted = get_predictions(network, xval)
    actual = torch.from_numpy(givenOut)#.float()
    
    # totCorrect = accuracy_metric(actual, predicted)
    
    # SR = totCorrect/len(xval[:,1])
    SR = torchmetrics.functional.accuracy(predicted, actual)
    loss = loss_function(predicted, actual)
    E = loss.item()

    if SR > 0.5:
        if len(pocket.weights) == 0:  # try:
            # print('--->')
            # print(nfc)
            pocket.weights = copy.deepcopy(network)
            pocket.sr = SR
            pocket.nfc = nfc
            pocket.s_epoch = epoch
            pocket.s_iteration = nfc

        # except IndexError:
        else:
            if pocket.sr < SR:
                # print(W)
                pocket.weights = copy.deepcopy(network)
                pocket.sr = SR
                pocket.nfc = nfc
                pocket.s_epoch = epoch
                pocket.s_iteration = nfc

        # print(SR)
        if SR > 110.85:  # percentage defined after decimal. <85% would be 0.85>
            doTerminate = True
        else:
            False

    N = np.size(inputVal, axis=0)  # number of cols    
    return doTerminate, SR, E, pocket  # PocketGoodWeights, doTerminate, SR, E, pocket

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i][0] == torch.round(predicted[i][0]).float():
			correct += 1
	return correct  


def get_predictions(network, xts):
    network.zero_grad()
    data_x = torch.from_numpy(xts).float()
    pred_y = network(data_x)
    #network.zero_grad()
    return pred_y
