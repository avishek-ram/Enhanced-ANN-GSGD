"""
 Programmer: Avishek Ram
 email: avishekram30@gmail.com
"""
from readData import readData
import numpy as np
import math
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os
import copy
from random import seed
from network import *
from getError import *
from validateSet import *
from printFinalResults import *
import torch
import torch.nn as nn
import sqlite3
from sqlite3 import Error
from main import *
from torchmetrics.utilities.checks import _input_format_classification
from data_layer import *

def GSGD_ANN_experiment(filePath):
    optims = ['ADADELTA'] #'RMSPROP','ADAGRAD', 'SGD', 'ADAM',

    connlite = connect()

    with connlite:
        for optim_idx in range(5):
            optim_name = optims[optim_idx]
            for run in range(50):
                # reading data, normalize and spliting into train/test
                NC, x, y, N, d, xts, yts = readData(filePath)
    
                #model parameters
                l_rate =   0.02001229522561126#0.019987676959698759#0.0001#0.0002314354244#9.309681215145698e-15#3.0952770286463463e-07#0.0003314354244#0.00011852732093870824#0.00010926827346753853 #0.0002814354245#0.0002216960781458557#0.0002314354244 #0.000885 #0.061 #0.00025 #0.5
                n_hiddenA = 30#36#29#50#36#4
                n_hiddenB = 5
                lamda =  1e-06#0.00014659309759736062#0.5964800918102662#0.06067045242012771#1e-05#0.6980844659683136 #1e-06#0.0001  #Lambda will be used for L2 regularizaion
                betas = (0.9, 0.999)
                beta = 0.9
                epsilon = 1e-8

                T = math.inf #number of batches to use in training. set to math.inf if all batches will be used in training
                is_guided_approach = True
                rho = 20
                versetnum = 5 #number of batches used for verification
                epochs = 20#27#15
                revisitNum = 15
                batch_size = 40#812#122#468#300#891#32

                #temporary experiment setup
                if(optim_name == 'RMSPROP'):
                    l_rate = 0.003098121514569
                    epochs = 5
                elif(optim_name == 'ADAGRAD'):
                    l_rate = 0.02465229341384616
                    epochs = 5
                elif(optim_name == 'SGD'):
                    l_rate =  0.18018122514569
                    epochs = 5
                elif(optim_name == 'ADAM'):
                    l_rate =  0.0035465229341384616
                    epochs = 5
                elif(optim_name == 'ADADELTA'):
                    l_rate =  0.2008229522561126
                    epochs = 20
                #end

                optim_params = l_rate, lamda, betas, beta, epsilon

                #Results Container
                GSGD_SRoverEpochs, GSGD_EoverEpochs, SGD_SRoverEpochs, SGD_EoverEpochs = [], [], [], []
                results_container = GSGD_SRoverEpochs, GSGD_EoverEpochs, SGD_SRoverEpochs, SGD_EoverEpochs

                #initialize both networks #should have the same initial weights
                network_GSGD = nn.Sequential(
                      nn.Linear(d, n_hiddenA),
                      nn.BatchNorm1d(n_hiddenA),
                      nn.Sigmoid(),
                      nn.Linear(n_hiddenA, n_hiddenB),
                      nn.BatchNorm1d(n_hiddenB),
                      nn.Sigmoid(),
                      nn.Linear(n_hiddenB, 1),
                      nn.Sigmoid()).to(device=device)
                optimizer_GSGD = get_optimizer(network_GSGD, name=optim_name, cache= optim_params)
                network_SGD = copy.deepcopy(network_GSGD)
                optimizer_SGD = get_optimizer(network_SGD, name=optim_name, cache= optim_params)

                experiment_param = optim_name, run+1, connlite

                #evaluate GSGD
                cache = is_guided_approach, rho, versetnum,epochs, revisitNum, N, network_GSGD, optimizer_GSGD, T, batch_size, NC
                evaluate_algorithm_experiment(x, y, xts, yts, cache, results_container, experiment_param)

                # evaluate SGD
                is_guided_approach = False
                cache = is_guided_approach, rho, versetnum,epochs, revisitNum, N, network_SGD, optimizer_SGD, T, batch_size, NC
                evaluate_algorithm_experiment(x, y, xts, yts, cache, results_container, experiment_param)

                #collction of final results and graphs
                generate_graphs_experiment(epochs, results_container, run+1, optim_name)


def generate_graphs_experiment(epochs, results_container, run, optim_name):
    GSGD_SRoverEpochs, GSGD_EoverEpochs, SGD_SRoverEpochs, SGD_EoverEpochs = results_container

    Epocperm = [ i+1 for i in range(epochs)]

    # Error Convergence of GSGD and SGD
    plt.figure()
    plt.plot(Epocperm, SGD_EoverEpochs, label=f'{optim_name} Error', linewidth=1)
    plt.plot(Epocperm, GSGD_EoverEpochs, 'r--', label=f'G{optim_name} Error', linewidth=1)

    plt.title(f'Error Convergence of G{optim_name} and {optim_name}')
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.legend(loc=2)

    plt.savefig('graphs/' + str(optim_name)+ '_run_'+ str(run) +'_error_convergence_general.png')

    #Success rate GSGD and SGD over Epochs General
    plt.figure()
    plt.plot(Epocperm, SGD_SRoverEpochs, label=f'{optim_name}', linewidth=1)
    plt.plot(Epocperm, GSGD_SRoverEpochs, 'r--', label=f'G{optim_name}', linewidth=1)

    plt.title(f'Classification Accuracy of G{optim_name} and {optim_name}')
    plt.xlabel("Epochs")
    plt.ylabel("Classification Accuracy")
    plt.legend(loc=2)

    plt.savefig('graphs/' + str(optim_name)+ '_run_'+ str(run) + '_success_rate_general.png')
  
def evaluate_algorithm_experiment(x, y, xts, yts, cache, results_container, experiment_param):
    loss_function = nn.MSELoss()
    StopTrainingFlag = False
    is_guided_approach, rho, versetnum, epochs, revisitNum, N, network, optimizer, T, batch_size, NC = cache
    
    GSGD_SRoverEpochs, GSGD_EoverEpochs, SGD_SRoverEpochs, SGD_EoverEpochs  = results_container

    #transform into tensors and setup dataLoader with mini batches
    my_dataset = TensorDataset(torch.Tensor(x), torch.Tensor(y))
    training_loader = DataLoader(my_dataset, batch_size=batch_size)

    #get mini batches
    x_batches, y_batches = [], []
    for input,labels in training_loader:
        x_batches.append(input)
        y_batches.append(labels)
    # end
    x_batches = np.array(x_batches)
    y_batches = np.array(y_batches)

    #transform Validation data
    xts = torch.Tensor(xts).to(device)
    yts = torch.Tensor(yts).to(device=device, dtype=torch.int)

     #Guided Training starts here
    if is_guided_approach:
        prev_error = math.inf #set initial error to very large number
        for epoch in range(epochs):
            getVerificationData = True
            verset_x, verset_response, avgBatchLosses, dataset_X, dataset_y, psi = [], [], [], [], [], []
            loopCount = -1
            revisit = False
            is_guided = False
            et = -1
            iteration = 0
            is_done = False
            batch_nums = len(x_batches)

            #shuffle batches
            shuffled_batch_indxs = np.random.permutation(batch_nums)
            new_X = copy.deepcopy(x_batches[shuffled_batch_indxs])
            new_y = copy.deepcopy(y_batches[shuffled_batch_indxs])

            #start training iterations
            while  not is_done and (not StopTrainingFlag):
                et +=1
                if et >= (batch_nums -1) or et >= T:
                    is_done = True
                #Set Verification Data at the beginning of epoch
                if getVerificationData:
                    versetindxs = shuffled_batch_indxs[:versetnum]
                    verset_x = np.array(new_X[versetindxs]) 
                    verset_response = np.array(new_y[versetindxs])
                    batch_nums = batch_nums - versetnum
                    new_X = np.delete(new_X, versetindxs, axis=0)
                    new_y = np.delete(new_y, versetindxs, axis=0)
                    getVerificationData = False  

                if not is_guided:
                    iteration = iteration + 1
                    loopCount = loopCount + 1
                    x_inst = new_X[et]
                    y_inst = new_y[et]
                    dataset_X.append(x_inst)
                    dataset_y.append(y_inst)

                    #1  train Network
                    train_network(network, x_inst.to(device= device), y_inst.to(device= device), loss_function, optimizer)

                    #now get verification data loss
                    veridxperms = np.random.permutation(versetnum)
                    veridxperm = veridxperms[0]
                    verloss = getErrorMSE(veridxperm, verset_x, verset_response, network, loss_function)
                    pos = 1
                    if verloss < prev_error:
                        pos = -1
                    
                    #Revist Previous Batches of Data and recalculate their
                    #losses only. WE DO NOT RE-UPDATE THE ENTIRE NETWORK WEIGHTS HERE. 
                    if revisit:
                        revisit_dataX = np.array(dataset_X)
                        revisit_dataY = np.array(dataset_y)

                        if loopCount == 1 or loopCount < revisitNum:
                            loopend = loopCount
                        else:
                            loopend = (revisitNum - 1) #In loops > 2, revisit previous 2 batches
                        currentBatchNumber = loopCount - 1
                        for i in range(loopend, loopCount, -1):
                            currentBatchNumber = currentBatchNumber - 1

                            #Reuse the layers outputs to compute loss of this revisit here => 
                            lossofrevisit = getErrorMSE(currentBatchNumber, revisit_dataX, revisit_dataY, network, loss_function)
                            
                            #previous batch was revisited and loss value is added into the array with previous batch losses
                            psi[currentBatchNumber] = np.append(psi[currentBatchNumber], ((-1 * pos) * (prev_error - lossofrevisit)))

                    #All batch error differences are collected into ?(psi).
                    current_batch_error = prev_error - verloss

                    psi.append(current_batch_error)

                    prev_error = verloss
                    revisit = True

                    #Check to see if its time for GSGD
                    if (iteration % rho) == 0:
                        is_guided = True
                else:
                    for k in range(loopCount):
                        avgBatchLosses = np.append(avgBatchLosses, np.mean(psi[k]))
                    
                    this_dataX = np.array(dataset_X)
                    this_dataY = np.array(dataset_y)

                    numel_avgBatch = len(avgBatchLosses)
                    avgBatchLosses_idxs = np.argsort(avgBatchLosses)[::-1]
                    avgBatchLosses = avgBatchLosses[avgBatchLosses_idxs] if len(avgBatchLosses_idxs) > 0 else []

                    min_repeat = min(rho/2, numel_avgBatch)
                    for r in range(int(min_repeat)):
                        if(avgBatchLosses[r] > 0):
                            guidedIdx = avgBatchLosses_idxs[r]
                            x_inst = this_dataX[guidedIdx]
                            y_inst = this_dataY[guidedIdx]
                            
                            train_network(network, x_inst.to(device= device), y_inst.to(device= device), loss_function, optimizer)

                            #Get Verification Data Loss
                            verIDX = np.random.permutation(versetnum)[0]

                            verLoss  = getErrorMSE(verIDX, verset_x, verset_response, network, loss_function)
                            prev_error = verLoss
                    
                    avgBatchLosses = []
                    psi = []
                    dataset_X = []
                    dataset_y = []
                    loopCount = -1
                    revisit = False
                    is_guided = False

                    #If an interrupt request has been made, break out of the epoch loop
                if StopTrainingFlag or is_done: 
                    break
        
            SR, E = validate(xts, network, yts, loss_function)
            # print('Epoch : %s' % str(epoch+1))
            # print('Accuracy: %s' % SR.item())
            # print('Error Rate: %s' % E)
            
            #Epoch Completes here
            GSGD_SRoverEpochs.append(SR.item())
            GSGD_EoverEpochs.append(E)

        experiment_results_final(xts, network, yts, loss_function, experiment_param, NC, type='guided')

    else: #not guided training
        print("Not Guided Training started")
        batch_nums = len(x_batches)

        for epoch in range(epochs):

            #shuffle batches
            shuffled_batch_indxs = np.random.permutation(batch_nums)
            new_X = x_batches[shuffled_batch_indxs]
            new_y = y_batches[shuffled_batch_indxs]

            for et in range(batch_nums):
                network.zero_grad()
                x_inst = new_X[et]
                y_inst = new_y[et]
                pred_y = network(x_inst.to(device= device))
                loss = loss_function(pred_y, y_inst.to(device= device))
                loss.backward()
                optimizer.step()

            iteration = 0
            SR, E = validate(xts, network, yts, loss_function)
            SGD_SRoverEpochs.append(SR.item())
            SGD_EoverEpochs.append(E)

            if(epoch == epochs-1):    
                print('Epoch : %s' % str(epoch+1))
                print('Accuracy: %s' % SR.item())
                print('Error Rate: %s' % E)
                experiment_results_final(xts, network, yts, loss_function, experiment_param, NC, type='original')

def experiment_results_final(inputVal, network, actual, loss_function, experiment_param, class_num,  type = ''):
    optim_name, run, connlite = experiment_param
    xval = inputVal
    
    random_colors = ['red', 'blue', 'green', 'yellow', 'orange','pink', 'purple', 'brown', 'black', 'grey' ]
    
    labels = [str(i) for i in range(class_num)]  #this is default
    
    #only used for diabetes dataset 2class and 3 class else use the above
    #labels = ["NO", "Readmitted"]
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
    for ind in range(len(labels)):
        plt.plot(fpr[ind].cpu().data.numpy(), tpr[ind].cpu().data.numpy(),'g--', color = random_colors[ind], label=labels[ind], marker='.', markersize='0.02')
    plt.title('ROC Curve')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=2)
    plt.savefig('graphs/'+ optim_name + '_run_'+ str(run) + type +'_roc_curve.png')

    #precision Recall Curve
    no_skill = len(actual[actual==1]) / len(actual)
    plt.figure()
    plt.plot([0.0, 1.0], [no_skill,no_skill], linestyle='--')
    for ind in range(len(labels)):
        plt.plot(recall_plot[ind].cpu().data.numpy(), precision_plot[ind].cpu().data.numpy(),  'g--', color = random_colors[ind], label=labels[ind], marker='.', markersize='0.02')
    plt.title('Precision Recall Curve')
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.legend(loc=2)
    plt.savefig('graphs/'+ optim_name + '_run_'+ str(run) + type +'_precision_recall_curve.png')

    #Confusion Matrix and Classification report
    print("\nClassification Report: " + type)
    preds_tranformed, actual_transformed, mode = _input_format_classification(preds=predicted, target= actual)
    print(metrics.classification_report(y_true = actual_transformed.cpu().data.numpy(), y_pred= preds_tranformed.cpu().data.numpy(), target_names=labels))
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = conf_matrix.cpu().data.numpy(), display_labels = labels)
    cm_display.plot()
    plt.title('Confusion Matrix')
    plt.savefig('graphs/'+ optim_name + '_run_'+ str(run) + type + '_confusion_matrix.png')
    #plt.show()

    with connlite:
            curlite = connlite.cursor()
            query = """INSERT INTO results
                (
                    optimizer,
                    type,
                    run,
                    sr,
                    loss
                )
                VALUES
                
                """
            query += f"""('{optim_name}','{type}', '{run}', '{accuracy.item()}', '{overall_E}')"""

            curlite.execute(query)
            connlite.commit()
            curlite.close()

if __name__ == '__main__':
    # root = tk.Tk()
    # root.withdraw()
    # file_path = filedialog.askopenfilename(
    #     initialdir=os.path.dirname(os.path.realpath(__file__))+'/data', filetypes=[('data files', '.data')])
    # print(file_path)
    # if(file_path == ''):
    #     print('File not found')

    #below ccode is only used in environment not supporting GUI/Tkinter, comment the above code wen using this
    #file_path = '/home/paperspace/Documents/Enhanced-ANN-GSGD/Enhanced-ANN-GSGD/data/diabetes_readmission_2class.data'
    file_path = 'C:/Users/avishek.ram/Documents/GitHub/Enhanced-ANN-GSGD/Enhanced-ANN-GSGD/data/diabetes_readmission_3class.data'
    GSGD_ANN_experiment(file_path)