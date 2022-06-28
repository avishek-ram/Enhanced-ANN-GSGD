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
    
    SR = totCorrect/len(xval[:,1]) * 100 
    print("Success rate: ", "{:.7f}".format(SR))

    return SR

def PrintFinalResults_updated(gr, PocketGoodWeights, inputVal, givenOut, printData, n_outputs):
        
    nfc = 0
    if bool(PocketGoodWeights):
        for i in range(len(PocketGoodWeights.weights), 0, -1):
            if len(PocketGoodWeights.weights) != 0:
                print('--------------------------')
                w = PocketGoodWeights.weights
                nfc = PocketGoodWeights.nfc
    else:
        w = PocketGoodWeights.weights
        nfc = 0
    
    # print
    if printData:
        xval = inputVal
        n_class = n_outputs

        predicted = get_predictions(PocketGoodWeights.weights, xval)
        actual = givenOut
        totCorrect = accuracy_metric(actual, predicted)
        confusion_matrix = np.zeros((n_class,n_class), dtype= np.int64)

        for row in range(len(xval)):
            the_unactivateds, the_activateds = forward_propagate(PocketGoodWeights.weights, xval[row])
            outputs = the_activateds[-1]
            outcome = np.argmax(outputs)
            confusion_matrix[givenOut[row,0], outcome] += 1
            
        #confusion matrix
        for row in range(n_class):
            truePositive = confusion_matrix[row,row]

            falseNegative = 0
            for i in range(n_class):
                if i != row:
                    falseNegative += confusion_matrix[row,i]

            falsePositive = 0
            for i in range(n_class):
                falsePositive += confusion_matrix[i, row]

            trueNegative = 0
            
            for i in range(n_class):
                if i != row:
                    trueNegative += confusion_matrix[i,i]
            
            SR = totCorrect/len(xval[:,1]) 
            print("Success rate: ", "{:.2f}".format(SR))
            print("On iteration: ", nfc)

            print('\nConfusion Matrix Details for class :' + str(row))
            print('True Positive: ', truePositive)
            print('True Negative: ', trueNegative)
            print('False Positive: ', falsePositive)
            print('False Negative: ', falseNegative)

            classificationAccuracy = (trueNegative+truePositive) / \
            (truePositive+trueNegative+falseNegative+falsePositive)

            misclassifiction = (falsePositive+falseNegative) / \
                (truePositive+trueNegative+falseNegative+falsePositive)

            recall = truePositive / truePositive+falseNegative

            precision = truePositive / truePositive+falsePositive

            specificity = trueNegative/trueNegative+falsePositive

            f1score = 2/((1/recall)+(1/precision))

            print('--Results-------  Class:'+ str(row))
            print('Classification Accuracy: ', classificationAccuracy)
            print('Misclassification: ', misclassifiction)
            print('Recall: ', recall)
            print('Precision: ', precision)
            print('Specificity: ', specificity)
            print('F1-score: ', f1score)
            print('----------------\n\n')

        return SR, nfc
            

