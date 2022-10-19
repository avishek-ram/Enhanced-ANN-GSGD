import numpy as np
from validateSet import *
import matplotlib.pyplot as plt
            
def print_results_final(inputVal, network, actual, loss_function, type = ''):
    xval = inputVal
    
    #get predicted value
    predicted = get_predictions(network, xval)
    
    accuracy = torchmetrics.functional.accuracy(predicted, actual)
    loss = loss_function(predicted, actual)
    overall_E = loss.item()
    recall = torchmetrics.functional.recall(preds=predicted, target=actual)
    precision = torchmetrics.functional.precision(preds=predicted, target=actual)
    specifity = torchmetrics.functional.specificity(preds=predicted, target=actual)
    f1score = torchmetrics.functional.f1_score(preds=predicted, target=actual)
    fpr, tpr, thresholds =  torchmetrics.functional.roc(preds=predicted, target=actual)
    precision_plot, recall_plot, thresholds_prc =  torchmetrics.functional.precision_recall_curve(preds=predicted, target=actual)

    print('--Results------'+ type)
    print('Classification Accuracy: ', accuracy.item())
    print('overall Error', overall_E)
    print('Recall: ', recall.item())
    print('Precision: ', precision.item())
    print('Specificity: ', specifity.item())
    print('F1-score: ', f1score.item())
    print('----------------\n\n')

    #ROC Curve
    plt.figure()
    plt.plot([0.0, 1.0], [0.0, 1.0], linestyle='--')
    plt.plot(fpr.cpu().data.numpy(), tpr.cpu().data.numpy(),'g--', label='ROC', marker='.', markersize='0.02')
    plt.title('ROC Curve')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('graphs/'+ type +'/roc_curve.png')
    
    #precision Recall Curve
    no_skill = len(actual[actual==1]) / len(actual)
    plt.figure()
    plt.plot([0.0, 1.0], [no_skill,no_skill], linestyle='--')
    plt.plot(recall_plot.cpu().data.numpy(), precision_plot.cpu().data.numpy(),'g--', label='Precision-Recall Curve', marker='.', markersize='0.02')
    plt.title('Precision Recall Curve')
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.savefig('graphs/'+ type +'/Precision_recall_curve.png')

def generate_graphs(epochs, results_container , T= 0, graph_first_epoch = False):
    GSGD_SRoverEpochs, GSGD_EoverEpochs, SGD_SRoverEpochs, SGD_EoverEpochs, singlepochSRGSGD, singlepochSRSGD = results_container

    Epocperm = [ i+1 for i in range(epochs)]
    singleepochperm = [ i+1 for i in range(len(singlepochSRGSGD))]

    # Error Convergence of GSGD and SGD
    plt.figure()
    plt.plot(Epocperm, SGD_EoverEpochs, label='SGD Error', linewidth=1)
    plt.plot(Epocperm, GSGD_EoverEpochs, 'r--', label='GSGD Error', linewidth=1)

    plt.title('Error Convergence of GSGD and SGD')
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.legend(loc=2)

    plt.savefig('graphs/error_convergence_general.png')

    #Success rate GSGD and SGD over Epochs General
    plt.figure()
    plt.plot(Epocperm, SGD_SRoverEpochs, label='SGD SR', linewidth=1)
    plt.plot(Epocperm, GSGD_SRoverEpochs, 'r--', label='GSGD SR', linewidth=1)

    plt.title('Classification Accuracy of GSGD and SGD')
    plt.xlabel("Epochs")
    plt.ylabel("Classification Accuracy")
    plt.legend(loc=2)

    plt.savefig('graphs/success_rate_general.png')

    # Success rate of GSGD and SGD for a single epoch # Works with T number of iterations only
    if graph_first_epoch:
        try:
            plt.figure()
            plt.plot(singleepochperm, singlepochSRGSGD, label='GSGD Classification Accuracy', linewidth=1)
            plt.plot(singleepochperm, singlepochSRSGD, 'r--', label='SGD Classification Accuracy', linewidth=1)

            plt.title('Success Rate of First Epoch')
            plt.xlabel("Epochs")
            plt.ylabel("Classification  Accuracy")
            plt.legend(loc=2)

            plt.savefig('graphs/classification_accuracy_single epoch.png')
        except:
            """error in Single Epoch Graph"""