import numpy as np
import math
from validate import *


def PrintFinalResults(gr, PocketGoodWeights, inputVal, givenOut, printData):

    nfc = 0
    w = list()
    if bool(PocketGoodWeights):
        for i in range(len(PocketGoodWeights.weights), 0, -1):
            if len(PocketGoodWeights.weights) != 0:
                print('--------------------------')
                w = PocketGoodWeights.weights
                nfc = PocketGoodWeights.nfc
    else:
        w = PocketGoodWeights.weights
        nfc = 0

    xval = inputVal
    #get predicted value
    predicted = get_predictions(w, xval)
    actual = givenOut
    totCorrect = accuracy_metric(actual, predicted)
        
    # SR code can be reused
    
    SR = totCorrect/len(xval[:,1]) * 100  # get num of cols
    print("Success rate: ", "{:.7f}".format(SR))
    print("On iteration: ", nfc)

    return SR, nfc


def PrintFinalResultsSGD(network_SGD, inputVal, givenOut):

    xval = inputVal

    #get predicted value
    predicted = get_predictions(network_SGD, xval)
    actual = givenOut
    totCorrect = accuracy_metric(actual, predicted)
        
    # SR code can be reused
    
    SR = totCorrect/len(xval[:,1]) * 100  # get num of cols
    print("Success rate: ", "{:.7f}".format(SR))

    return SR