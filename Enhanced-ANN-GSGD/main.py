"""
 Programmer: Avishek Anishkar Ram
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
from torch.utils.data import DataLoader, TensorDataset

seed(1)
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 
np.warnings.filterwarnings('ignore', category=FutureWarning) 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def GSGD_ANN(filePath):
    # reading data, normalize and spliting into train/test
    NC, x, y, N, d, xts, yts = readData(filePath)
    
    #model parameters
    l_rate =    0.0035465229341384616
    n_hiddenA = 30
    n_hiddenB = 5
    lamda =  1e-06
    betas = (0.9, 0.999)
    beta = 0.9
    epsilon = 1e-8

    T = math.inf #number of batches to use in training. set to math.inf to use all batches
    is_guided_approach = True
    rho = 20
    versetnum = 5 
    epochs = 5
    revisitNum = 15
    batch_size = 40

    optim_params = l_rate, lamda, betas, beta, epsilon
    optims = ['SGD', 'ADAM', 'ADADELTA', 'RMSPROP', 'ADAGRAD']
    optim_name = optims[1]

    #Results Container
    GSGD_SRoverEpochs, GSGD_EoverEpochs, SGD_SRoverEpochs, SGD_EoverEpochs = [], [], [], []
    results_container = GSGD_SRoverEpochs, GSGD_EoverEpochs, SGD_SRoverEpochs, SGD_EoverEpochs

    #initialize both networks #networks should have the same initial weights
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

    #evaluate GSGD
    cache = is_guided_approach, rho, versetnum,epochs, revisitNum, N, network_GSGD, optimizer_GSGD, T, batch_size
    evaluate_algorithm(x, y, xts, yts, cache, results_container)

    # evaluate SGD
    is_guided_approach = False
    cache = is_guided_approach, rho, versetnum,epochs, revisitNum, N, network_SGD, optimizer_SGD, T, batch_size
    evaluate_algorithm(x, y, xts, yts, cache, results_container)

    #collection of final results and graphs
    generate_graphs(epochs, results_container)
    
def evaluate_algorithm(x, y, xts, yts, cache, results_container):
    loss_function = nn.MSELoss()
    StopTrainingFlag = False
    is_guided_approach, rho, versetnum, epochs, revisitNum, N, network, optimizer, T, batch_size = cache
    
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
            print('Epoch : %s' % str(epoch+1))
            print('Accuracy: %s' % SR.item())
            print('Error Rate: %s' % E)

            #Epoch Completes here
            GSGD_SRoverEpochs.append(SR.item())
            GSGD_EoverEpochs.append(E)

        print_results_final(xts, network, yts, loss_function, type='guided')

    else: #not guided training
        print("\n\n-----Not Guided Training------")
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
