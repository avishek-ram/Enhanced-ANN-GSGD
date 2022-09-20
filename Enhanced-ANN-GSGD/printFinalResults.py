import numpy as np
from validate import *
import matplotlib.pyplot as plt
            
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