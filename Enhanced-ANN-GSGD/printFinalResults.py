import numpy as np
import math
from validate import *
import matplotlib.pyplot as plt


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

            recall = truePositive / (truePositive+falseNegative)

            precision = truePositive / (truePositive+falsePositive)

            specificity = trueNegative/(trueNegative+falsePositive)

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
            
def print_results_final(inputVal, network, givenOut, loss_function, type = '' , pocket = None):
    xval = inputVal
    
    #get predicted value
    predicted = get_predictions(network, xval)
    actual = torch.from_numpy(givenOut)#.float()
    
    accuracy = torchmetrics.functional.accuracy(predicted, actual)
    loss = loss_function(predicted, actual)
    overall_E = loss.item()
    recall = torchmetrics.functional.recall(preds=predicted, target=actual)
    precision = torchmetrics.functional.precision(preds=predicted, target=actual)
    specifity = torchmetrics.functional.specificity(preds=predicted, target=actual)
    f1score = torchmetrics.functional.f1_score(preds=predicted, target=actual)
    fpr, tpr, thresholds =  torchmetrics.functional.roc(preds=predicted, target=actual)

    print('--Results------'+ type)
    print('Classification Accuracy: ', accuracy.item())
    print
    print('Recall: ', recall.item())
    print('Precision: ', precision.item())
    print('Specificity: ', specifity.item())
    print('F1-score: ', f1score.item())
    print('----------------\n\n')

    plt.figure()
    plt.plot(fpr, tpr,'b--', label='ROC', linewidth=1)
    plt.title('ROC Curve')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('graphs/'+ type +'/roc_curve.png')

def generate_graphs(epochs, results_container):
    GSGD_SRoverEpochs, GSGD_SRpocketoverEpochs, GSGD_EoverEpochs, GSGD_EpocketoverEpochs, SGD_SRoverEpochs, SGD_EoverEpochs = results_container

    Epocperm = [ i+1 for i in range(epochs)]

    # Error Convergence of GSGD and SGD  - GSGD is not pocket Best
    plt.figure()
    plt.plot(Epocperm, SGD_EoverEpochs, label='SGD Error', linewidth=1)
    plt.plot(Epocperm, GSGD_EoverEpochs, 'r--', label='GSGD Error', linewidth=1)

    plt.title('Error Convergence of GSGD and SGD')
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.legend(loc=2)

    plt.savefig('graphs/error_convergence_general.png')

    # Error Convergence of GSGD and SGD  - GSGD is pocket Best
    plt.figure()
    plt.plot(Epocperm, SGD_EoverEpochs, label='SGD Error', linewidth=1)
    plt.plot(Epocperm, GSGD_EpocketoverEpochs, 'r--', label='GSGD Error', linewidth=1)

    plt.title('Error Convergence of GSGD and SGD')
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.legend(loc=2)

    plt.savefig('graphs/error_convergence_pocketedGSGD.png')

    #Success rate GSGD and SGD over Epochs General
    plt.figure()
    plt.plot(Epocperm, SGD_SRoverEpochs, label='SGD SR', linewidth=1)
    plt.plot(Epocperm, GSGD_SRoverEpochs, 'r--', label='GSGD SR', linewidth=1)

    plt.title('Success Rate of GSGD and SGD')
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.legend(loc=2)

    plt.savefig('graphs/success_rate_general.png')

    #Success rate GSGD and SGD over Epochs  - GSGD is pocket Best
    plt.figure()
    plt.plot(Epocperm, SGD_SRoverEpochs, label='SGD SR', linewidth=1)
    plt.plot(Epocperm, GSGD_SRpocketoverEpochs, 'r--', label='GSGD SR', linewidth=1)

    plt.title('Success Rate of GSGD and SGD')
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.legend(loc=2)

    plt.savefig('graphs/success_rate_PocketedGSGD.png')