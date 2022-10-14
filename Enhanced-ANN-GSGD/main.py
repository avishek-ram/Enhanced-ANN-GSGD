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
from validate import *
from printFinalResults import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def GSGD_ANN(filePath):
    # reading data, normalize and spliting into train/test
    NC, x, y, N, d, xts, yts = readData(filePath)

    seed(1)
    
    #model parameters
    l_rate =  0.00010926827346753853 #0.0002814354245#0.0002216960781458557#0.0002314354244 #0.000885 #0.061 #0.00025 #0.5
    n_hidden = 36 #4
    lamda =  0.6980844659683136 #1e-06#0.0001  #Lambda will be used for L2 regularizaion
    betas = (0.9, 0.999)
    beta = 0.9
    epsilon = 1e-8

    optim_params = l_rate, lamda, betas, beta, epsilon
    optims = ['SGD', 'ADAM', 'ADADELTA', 'RMSPROP', 'ADAGRAD']
    
    optim_name = optims[0]

    #Results Container
    GSGD_SRoverEpochs = []
    GSGD_EoverEpochs = []

    SGD_SRoverEpochs = []
    SGD_EoverEpochs = []

    singlepochSRGSGD = []
    singlepochSRSGD = []

    results_container = GSGD_SRoverEpochs, GSGD_EoverEpochs, SGD_SRoverEpochs, SGD_EoverEpochs, singlepochSRGSGD, singlepochSRSGD

    #initialize both networks #should have the same initial weights
    network_GSGD = nn.Sequential(
                      nn.Linear(d, n_hidden),
                      nn.Sigmoid(),
                      nn.Linear(n_hidden, 1),
                      nn.Sigmoid())
    optimizer_GSGD = get_optimizer(network_GSGD, name=optim_name, cache= optim_params)
    network_SGD = copy.deepcopy(network_GSGD)
    optimizer_SGD = get_optimizer(network_SGD, name=optim_name, cache= optim_params)


    # evaluate algorithm GSGD
    T = 1000
    is_guided_approach = True
    rho = 7
    versetnum = 5
    epochs = 30
    revisitNum = 3
    cache = is_guided_approach, rho, versetnum,epochs, revisitNum, N, network_GSGD, optimizer_GSGD, T
    evaluate_algorithm(back_propagation, x, y, xts, yts , l_rate, n_hidden, d, NC, N, filePath, lamda, cache, results_container)

    # evaluate algorithm SGD
    is_guided_approach = False
    cache = is_guided_approach, rho, versetnum,epochs, revisitNum, N, network_SGD, optimizer_SGD, T
    evaluate_algorithm(back_propagation, x, y, xts, yts , l_rate, n_hidden, d, NC, N, filePath, lamda, cache, results_container)

    #collction of final results and graphs
    generate_graphs(epochs, results_container, T)
        
def evaluate_algorithm(algorithm, x, y, xts, yts , l_rate, n_hidden, d, NC, N, filePath, lamda, cache , results_container):

    algorithm(x, y, xts, yts, l_rate, n_hidden, d, NC, N, filePath, lamda, cache, results_container)
    
def back_propagation(x, y, xts, yts, l_rate, n_hidden, n_inputs, n_outputs, N, filePath, lamda, cache, results_container):
    loss_function = nn.MSELoss()
    StopTrainingFlag = False
    is_guided_approach, rho, versetnum, epochs, revisitNum, N, network, optimizer, T = cache
    
    GSGD_SRoverEpochs, GSGD_EoverEpochs, SGD_SRoverEpochs, SGD_EoverEpochs, singlepochSRGSGD, singlepochSRSGD  = results_container

    #start epoch
    if is_guided_approach:
        prev_error = math.inf #set initial error to very large number
        for epoch in range(epochs):
            epoch_sr =  0.0
            epoch_E = math.inf
            getVerificationData = True
            verset_x = []
            verset_response = []
            avgBatchLosses = []
            loopCount = -1
            psi = []
            revisit = False
            is_guided = False
            shuffled_order = np.random.permutation(N-1)
            et = -1
            updated_N = N
            new_X = copy.deepcopy(x)
            new_y = copy.deepcopy(y)
            dataset_X = []
            dataset_y = []
            iteration = 0
            is_done = False
            #start training iterations
            while  not is_done and (not StopTrainingFlag):
                et +=1
                if et >= (updated_N -1) or et >= T:
                    is_done = True
                #Set Verification Data at the beginning of epoch
                if getVerificationData:
                    indxToRemove = []
                    for vercount in range(versetnum):
                        indx = shuffled_order[vercount]
                        indxToRemove.append(indx)
                        x_inst = new_X[[indx], :]
                        y_inst = new_y[[indx], :]
                        verset_x.append(x_inst)
                        verset_response.append(y_inst)
                    updated_N = N - versetnum
                    verset_x  = np.array(verset_x).reshape(versetnum, n_inputs)
                    verset_response = np.array(verset_response).reshape(versetnum, 1)
                    #update new x and y
                    new_X = np.delete(new_X, indxToRemove, axis=0)
                    new_y = np.delete(new_y, indxToRemove, axis=0)
                    getVerificationData = False
                
                if not is_guided:
                    iteration = iteration + 1
                    loopCount = loopCount + 1
                    x_inst = new_X[[et], :]
                    y_inst = new_y[[et], :]
                    dataset_X.append(x_inst)
                    dataset_y.append(y_inst)

                    #1  train Network
                    train_network(network, x_inst, y_inst, l_rate, n_outputs, lamda, n_inputs, loss_function, optimizer)
                   

                    #now get verification data loss
                    veridxperms = np.random.permutation(versetnum-1)
                    veridxperm = veridxperms[0]
                    ver_x = verset_x[[veridxperm], :]
                    ver_y = verset_response[[veridxperm], :]
                    #run foward propagation
                    # verloss = getErrorCrossEntropy(veridxperm, verset_x, verset_response, network, n_outputs)
                    verloss = getErrorMSE(veridxperm, verset_x, verset_response, network, loss_function)
                    #calculate loss of this verification instance  => verloss
                    pos = 1
                    if verloss < prev_error:
                        pos = -1
                    
                    #Revist Previous Batches of Data and recalculate their
                    #losses only. WE DO NOT RE-UPDATE THE ENTIRE NETWORK WEIGHTS HERE. 
                    if revisit:
                        revisit_dataX = np.array(dataset_X).reshape(len(dataset_X), n_inputs)
                        revisit_dataY = np.array(dataset_y).reshape(len(dataset_y), 1)

                        if loopCount == 1 or loopCount < revisitNum:
                            loopend = loopCount
                        else:
                            loopend = (revisitNum - 1) #In loops > 2, revisit previous 2 batches
                        currentBatchNumber = loopCount - 1
                        for i in range(loopend, loopCount, -1):
                            currentBatchNumber = currentBatchNumber - 1

                            #forward propagte revisit_x
                            #Reuse the layers outputs to compute loss of this revisit here => 
                            #lossofrevisit = getErrorCrossEntropy(currentBatchNumber, revisit_dataX, revisit_dataY, network, n_outputs)
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
                    
                    this_dataX = np.array(dataset_X).reshape(len(dataset_X), n_inputs)
                    this_dataY = np.array(dataset_y).reshape(len(dataset_y), 1)

                    numel_avgBatch = len(avgBatchLosses)
                    avgBatchLosses_idxs = np.argsort(avgBatchLosses)[::-1]
                    avgBatchLosses = avgBatchLosses[avgBatchLosses_idxs] if len(avgBatchLosses_idxs) > 0 else []

                    min_repeat = min(rho/2, numel_avgBatch)
                    for r in range(int(min_repeat)):
                        if(avgBatchLosses[r] > 0):
                            guidedIdx = avgBatchLosses_idxs[r]
                            x_inst = this_dataX[[guidedIdx],:]
                            y_inst = this_dataY[[guidedIdx],:]
                            
                            train_network(network, x_inst, y_inst, l_rate, n_outputs, lamda, n_inputs, loss_function, optimizer)

                            #Get Verification Data Loss
                            verIDX = np.random.permutation(versetnum)[0]
                            verx = verset_x[[verIDX], :]
                            very = verset_response[[verIDX], :]

                            # verLoss  = getError(verIDX, verset_x, verset_response, network, n_outputs)
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
        
                doTerminate, SR, E = validate(
                            xts, network, yts, iteration, n_outputs, epoch+1, loss_function)

                if(SR.item() > epoch_sr):
                    epoch_sr = SR.item()
                    epoch_E = E
                print('Epoch : %s' % str(epoch+1))
                print('Success Rate: %s' % SR.item())
                print('Error Rate: %s' % E)
                if epoch == 0:
                    singlepochSRGSGD.append(SR.item())


            #Epoch Completes here
            GSGD_SRoverEpochs.append(SR.item())
            GSGD_EoverEpochs.append(E)

        print_results_final(xts, network, yts, loss_function, type='guided')

    else: #not guided training in mini batches
        print("Not Guided Training started")
        shuffled_idxs = np.random.permutation(N-1)

        tensor_x = torch.Tensor(x[shuffled_idxs])
        tensor_y = torch.Tensor(y[shuffled_idxs])

        my_dataset = TensorDataset(tensor_x, tensor_y)
        training_loader = DataLoader(my_dataset, batch_size=77, shuffle=True)

        for epoch in range(epochs):
            
            for input,labels in training_loader:
                network.zero_grad()
                pred_y = network(input)
                loss = loss_function(pred_y, labels)
                loss.backward()
                optimizer.step()


            # iteration = 0
            # for idx in shuffled_idxs:
            #     iteration += 1
            #     network.zero_grad()
            #     data_x = torch.from_numpy(x[[idx],:]).float()
            #     pred_y = network(data_x)
            #     data_y = torch.from_numpy(y[[idx],:]).float() #row_label should numpy array
            #     loss = loss_function(pred_y, data_y)
            #     loss.backward()

            #     optimizer.step()
                                
            #     if(epoch == 0):
            #         doTerminate, SR, E = validate(
            #                 xts, network, yts, iteration, n_outputs, epoch+1, loss_function)
            #         singlepochSRSGD.append(SR.item())

            #     if(iteration >= T):
            #         break
            
            iteration = 0
            doTerminate, SR, E = validate(
                            xts, network, yts, iteration, n_outputs, epoch+1, loss_function)

            SGD_SRoverEpochs.append(SR.item())
            SGD_EoverEpochs.append(E)

            if(epoch == epochs-1):    
                print('Epoch : %s' % str(epoch+1))
                print('Success Rate: %s' % SR.item())
                print('Error Rate: %s' % E)
                print_results_final(xts, network, yts, loss_function, type='original')

if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        initialdir=os.path.dirname(os.path.realpath(__file__))+'/data', filetypes=[('data files', '.data')])
    print(file_path)
    if(file_path == ''):
        print('File not found')
    GSGD_ANN(file_path)
