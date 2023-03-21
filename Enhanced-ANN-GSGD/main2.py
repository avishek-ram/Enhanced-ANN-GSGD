from readData import readData
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os
import copy
from math import exp
from random import seed
from propagation import *
from getError import *
from collectInconsistentInstances import collectInconsistentInstances
from extractConsistentInstances import extractConsistentInstances
from validate import *
from printFinalResults import *

def GSGD_ANN(filePath):
    # reading data, normalize and spliting into train/test
    NC, x, y, N, d, xts, yts = readData(filePath)

    seed(1)
    
    #model parameters
    l_rate = 0.5
    n_hidden = 2
    lamda = 0.000001  #Lambda will be used for regularizaion
    
    #initialize both networks #should have the same initial weights
    network_GSGD = initialize_network(n_hidden, d , NC)
    network_SGD = copy.deepcopy(network_GSGD)

    # evaluate algorithm GSGD
    is_guided_approach = True
    rho = 7
    versetnum = 10
    epochs = 15
    revisitNum = 8
    cache = is_guided_approach, rho, versetnum,epochs, revisitNum, N, network_GSGD
    evaluate_algorithm(back_propagation, x, y, xts, yts , l_rate, n_hidden, d, NC, N, filePath, lamda, cache)

    # evaluate algorithm SGD
    is_guided_approach = False
    epochs = 15 #Different Epoch values can be used since GSGD has better convergence
    cache = is_guided_approach, rho, versetnum,epochs, revisitNum, N, network_SGD
    evaluate_algorithm(back_propagation, x, y, xts, yts , l_rate, n_hidden, d, NC, N, filePath, lamda, cache)
        
def evaluate_algorithm(algorithm, x, y, xts, yts , l_rate, n_hidden, d, NC, N, filePath, lamda, cache):   
    algorithm(x, y, xts, yts, l_rate, n_hidden, d, NC, N, filePath, lamda,cache)
    
def back_propagation(x, y, xts, yts, l_rate, n_hidden, n_inputs, n_outputs, N, filePath, lamda, cache):
    StopTrainingFlag = False
    is_guided_approach, rho, versetnum, epochs, revisitNum, N, network = cache
    T = 500

    class PGW:
        weights = list()
        nfc = 00
        sr = 0
        s_epoch = 0
        s_iteration = 0
    
    pocket = PGW()
    pocketSGD = PGW()
    
    #start epoch
    if is_guided_approach:
        prev_error = math.inf #set initial error to very large number
        for epoch in range(epochs):
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
            while  not is_done and (not StopTrainingFlag):  # might have to remove stopTraining flag, matlab code
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
                        verset_x.append(x_inst)#np.append(verset_x, x_inst, axis=0)
                        verset_response.append(y_inst)#np.append(verset_response, y_inst, axis=0)
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
                    train_network(network, x_inst, y_inst, l_rate, n_outputs, lamda, n_inputs)
                    #2 get loss
                    
                    #3 get gradients, regularize if needed
                    #4 update learnable parameters
                    #5 update weights of network #maybe we will only have to calculate mew weights and not update the  network
                    #6 update summary (optional)

                    #now get verification data loss
                    veridxperms = np.random.permutation(versetnum-1)
                    veridxperm = veridxperms[0]
                    ver_x = verset_x[[veridxperm], :]
                    ver_y = verset_response[[veridxperm], :]
                    #run foward propagation
                    verloss = getErrorCrossEntropy(veridxperm, verset_x, verset_response, network, n_outputs)
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
                            lossofrevisit = getErrorCrossEntropy(currentBatchNumber, revisit_dataX, revisit_dataY, network, n_outputs)
                            
                            #previous batch was revisited and loss value is added into the array with previous batch losses
                            #np.append(psi[currentBatchNumber, :], lossofrevisit) old code
                            psi[currentBatchNumber] = np.append(psi[currentBatchNumber], ((-1 * pos) * (prev_error - lossofrevisit)))  #have to recheck the axis might need to be specified

                    #All batch error differences are collected into ?(psi).
                    current_batch_error = prev_error - verloss
                    # if (loopCount > 0) and (len(psi) >= loopCount):
                    #     psi[loopCount] = current_batch_error
                    # else:
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
                            #forward propagate 
                            #calculate new gradients
                            #miniBatchLoss = netloss
                            #gradients rgularized
                            # update weights
                            #update sunmmary
                            train_network(network, x_inst, y_inst, l_rate, n_outputs, lamda, n_inputs)

                            #Get Verification Data Loss
                            verIDX = np.random.permutation(versetnum)[0]
                            verx = verset_x[[verIDX], :]
                            very = verset_response[[verIDX], :]
                            # forward propagate 
                            verLoss  = getError(verIDX, verset_x, verset_response, network, n_outputs)
                            prev_error = verLoss
                    
                    avgBatchLosses = []
                    psi = []
                    dataset_X = []
                    dataset_y = []
                    loopCount = -1
                    revisit = False
                    is_guided = False

                    #learnRate #= new learn rate if needed to update
                    #If an interrupt request has been made, break out of the epoch loop
                if StopTrainingFlag: 
                    break
        
                doTerminate, SR, E, pocket = validate(
                            xts, network, yts, iteration, pocket, n_outputs, epoch+1)

                print('Epoch : %s' % str(epoch+1))
                print('Success Rate: %s' % SR)
                print('Error Rate: %s' % E)
    
        # compute Finish Summary
        doTerminate, SR, E, pocket = validate(
                            xts, network, yts, iteration, pocket, n_outputs, epoch+1)
        
        print('Epoch : %s' % str(epoch+1))
        print('Success Rate: %s' % SR)
        print('Error Rate: %s' % E)

        #Final summary
        print('Success Rate of best weights: %s' % pocket.sr)
        print('using pocket weights ...')
        doTerminate, SR, E, pocket = validate(
                            xts, pocket.weights, yts, iteration, pocket, n_outputs, epoch+1)
        
        print('Epoch : %s' % str(epoch+1))
        print('Success Rate: %s' % SR)
        print('Error Rate: %s' % E)

    else: #not guided training
        print("Not Guided Training started")
        for epoch in range(epochs):
            shuffled_idxs = np.random.permutation(N-1)
            iteration = 0
            for idx in shuffled_idxs:
                iteration += 1
                train_network(network, x[[idx],:], y[[idx],:], l_rate, n_outputs, lamda, n_inputs)
                doTerminate, sgdSR, sgdE, pocketSGD = validate(
                            xts, network, yts, iteration, pocketSGD, n_outputs, epoch+1)
                
                if(iteration >= T):
                    break

            # SGDdoTerminate, sgdSR, sgdE = validateSGD(
            #         xts, network, yts, n_outputs)
            
            if(epoch == epochs-1):
                doTerminate, SR, E, pocketSGD = validate(
                            xts, network, yts, iteration, pocket, n_outputs, epoch+1)
        
                print('Epoch : %s' % str(epoch+1))
                print('Success Rate: %s' % SR)
                print('Error Rate: %s' % E)

if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        initialdir=os.path.dirname(os.path.realpath(__file__))+'/data', filetypes=[('data files', '.data')])
    print(file_path)
    if(file_path == ''):
        print('File not found')
    GSGD_ANN(file_path)