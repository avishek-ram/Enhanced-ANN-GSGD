import numpy as np
from validateSet import *
import matplotlib.pyplot as plt
from sklearn import metrics
from torchmetrics.utilities.checks import _input_format_classification
            
def print_results_final(inputVal, network, actual, loss_function, class_num, type = ''):
    xval = inputVal
    
    #only used for diabetes dataset 2class and 3 class else set to None
    labels = None
    #labels = ["Not", "Readmitted"]
    #labels = ["NO", "<30", ">30"]
    #end

    #get predicted value
    predicted = get_predictions(network, xval)
    
    accuracy = torchmetrics.functional.accuracy(predicted, actual)
    loss = loss_function(predicted, actual)
    overall_E = loss.item()
    specifity = torchmetrics.functional.specificity(preds=predicted, target=actual)
    fpr, tpr, thresholds =  torchmetrics.functional.roc(preds=predicted, target=actual, num_classes=class_num)
    precision_plot, recall_plot, thresholds_prc =  torchmetrics.functional.precision_recall_curve(preds=predicted, target=actual, num_classes=class_num)
    conf_matrix = torchmetrics.functional.confusion_matrix(preds=predicted, target=actual, num_classes= class_num)

    print('\n\n--Results------'+ type)
    print('Classification Accuracy: ', accuracy.item())
    print('overall Error', overall_E)
    print('----------------')

    #ROC Curve
    plt.figure()
    plt.plot([0.0, 1.0], [0.0, 1.0], linestyle='--')
    plt.plot(fpr[0].cpu().data.numpy(), tpr[0].cpu().data.numpy(),'g--', label='ROC', marker='.', markersize='0.02')
    plt.title('ROC Curve')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('graphs/'+ type +'/roc_curve.png')
    
    #precision Recall Curve
    no_skill = len(actual[actual==1]) / len(actual)
    plt.figure()
    plt.plot([0.0, 1.0], [no_skill,no_skill], linestyle='--')
    plt.plot(recall_plot[1].cpu().data.numpy(), precision_plot[1].cpu().data.numpy(),'g--', label='Precision-Recall Curve', marker='.', markersize='0.02')
    plt.title('Precision Recall Curve')
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.savefig('graphs/'+ type +'/Precision_recall_curve.png')

    #Confusion Matrix and Classification report
    print("\nClassification Report: " + type)
    preds_tranformed, actual_transformed, mode = _input_format_classification(preds=predicted, target= actual)
    print(metrics.classification_report(y_true = actual_transformed.cpu().data.numpy(), y_pred= preds_tranformed.cpu().data.numpy(), target_names=labels))
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = conf_matrix.cpu().data.numpy(), display_labels = labels)
    cm_display.plot()
    plt.title('Confusion Matrix')
    plt.savefig('graphs/' + type + '/confusion_matrix.png')
    #plt.show()

def generate_graphs(epochs, results_container):
    GSGD_SRoverEpochs, GSGD_EoverEpochs, SGD_SRoverEpochs, SGD_EoverEpochs = results_container
    Epocperm = [ i+1 for i in range(epochs)]

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
    plt.plot(Epocperm, SGD_SRoverEpochs, label='SGD Accuracy', linewidth=1)
    plt.plot(Epocperm, GSGD_SRoverEpochs, 'r--', label='GSGD Accuracy', linewidth=1)

    plt.title('Classification Accuracy of GSGD and SGD')
    plt.xlabel("Epochs")
    plt.ylabel("Classification Accuracy")
    plt.legend(loc=2)

    plt.savefig('graphs/success_rate_general.png')