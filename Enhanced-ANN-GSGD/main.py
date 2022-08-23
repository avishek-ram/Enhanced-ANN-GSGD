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
from getError import getError
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
    n_epoch = 30
    n_hidden = 1
    lamda = 0.0001  #Lambda will be used for regularizaion
    verfset =  0.0001  #percentage of dataset to use for verification; Decrease this value for large datasets
    
    # evaluate algorithm
    evaluate_algorithm(back_propagation, x, y, xts, yts , l_rate, n_hidden, d, NC, N, n_epoch, filePath, lamda, verfset)
        
def evaluate_algorithm(algorithm, x, y, xts, yts , l_rate, n_hidden, d, NC, N, n_epoch, filePath, lamda, verfset):
    algorithm(x, y, xts, yts, l_rate, n_hidden, d, NC, N, n_epoch, filePath, lamda, verfset)
    
def back_propagation(x, y, xts, yts, l_rate, n_hidden, n_inputs, n_outputs, N, n_epoch, filePath, lamda, verfset):
     
    #Start GSGD and SGD implemenattion here
    ropeTeamSz = 10  # this is rho. neighborhood size => increase rho value when the dataset is very noisy.
    pe = math.inf
    t = 0
    idx = np.array(np.random.permutation(N-1))
    NFC = 0
    SGDnfc = 0
    et = -1
    E = math.inf
    best_E = math.inf
        
    class PGW:
        weights = list()
        nfc = 0
        sr = 0

    pocket = PGW()
    
    plotE = []
    plotEgens = []
    plotEout = []
    plotEin = []
    PlotEoutSR = []

    plotEoutSRsgd = []
    plotESGD = []
    SGDEout = []
    SGDEin = []
    plotEgensSGD = []
    SRovertime = []
    SRSGDOvertime = []
    GSGDError = []
    SGDError = []

    bPlot = True
    
    #initialize network here / inital weights
    network_SGD = initialize_network(n_hidden, n_inputs , n_outputs)
    network_GSGD = copy.deepcopy(network_SGD) # when starting both should have the same weights
    for ep in range(n_epoch):
        idxs = idx
        #reseting the variables here for this epoch
        NFC = 0
        SGDnfc = 0
        plotE = []
        plotEgens = []
        plotEout = []
        plotEin = []
        PlotEoutSR = []

        plotEoutSRsgd = []
        plotESGD = []
        SGDEout = []
        SGDEin = []
        plotEgensSGD = []
        #end reset
        et = -1
        T = 69000#3681
        for t in range(T):      
            et = et + 1
            # if not idx[et]:
            if idx.size == 0:
                idx = np.random.permutation(N-1)
                et = 0
                idxs = idx  
            
            curIdx = idx[et]
            
            # update weights for the iteration
            train_network(network_GSGD, x[[curIdx],:], y[[curIdx],:], l_rate, n_outputs, lamda, n_inputs)
            NFC = NFC + 1
            #end
            
            er = np.random.permutation(N)
            verfsetsize = round(N*verfset)
            er = np.array([er[0:verfsetsize]])
                    
            #get SGD weights
            curIdxs = idxs[0]
            idxs = np.delete(idxs, 0, axis=0)
            train_network(network_SGD, x[[curIdxs],:], y[[curIdxs],:], l_rate, n_outputs, lamda, n_inputs)
            SGDnfc = SGDnfc + 1
            #end
            
            ve = 0
            eSGD = 0
            
            #with the initialized weight we will now try to get averaged error value of  random rows for both
            for k in er[0]:
                ve = ve + getError(k, x, y, network_GSGD, n_outputs)
                eSGD = eSGD + getError(k, x, y, network_SGD, n_outputs)
            
            ve = ve/len(er[0])
            eSGD = eSGD/len(er[0])
            
            # update best_E new code draft
            if ve < best_E:
                best_E = ve
            
            # collect inconsistent instances
            omPlusScore, omPlusLevel, omMinuScore, omMinusLevel, tmpGuided = collectInconsistentInstances(
                idx, x, y, network_GSGD, ropeTeamSz, best_E, n_outputs)
            
            # extract consistent instances
            consistentIdx = extractConsistentInstances(
                ve, best_E, omPlusScore, omPlusLevel, omMinuScore, omMinusLevel)
            
            # further refinement
            for cI in range(len(consistentIdx)):
                curId = consistentIdx[cI]  # set index to consistentIdx value
                # W, s, r, gradHist, Whistory, mAdam, vAdam = SGDvariation(t, x[:,[cI]], y[:,[cI]], W, eta, 'cross-entropy','canonical', s,r,gradHist, Whistory, mAdam, vAdam)
                train_network(network_GSGD, x[[curId],:], y[[curId],:], l_rate, n_outputs, lamda, n_inputs)
                NFC = NFC+1
                
            if t % ropeTeamSz == 0 or t % tmpGuided == 0:  # or et > idx.size:
                idx = np.delete(idx, np.s_[0:tmpGuided], axis=0)
                et = -1
            
            else:
                # tmpVals = np.setdiff1d(idx[0:tmpGuided], idx[consistentIdx])
                tmpVals = np.setdiff1d(idx[0:tmpGuided], idx[consistentIdx])
                idxArray = np.arange(tmpGuided)
                inconsistentIdx = np.setdiff1d(idxArray, consistentIdx)

                # remove consistent idx from list to only have inconsistent idx... 0 is an index** 
                idx = np.delete(idx, inconsistentIdx, axis=0)

                idx = np.r_[idx, tmpVals]
            
            # Plot section
            if t % 5000 == 0 or t == T:

                plotE = np.append(plotE, ve)
                plotEgens = np.append(plotEgens, t)

                doTerminate, SR, E, pocket = validate(
                    xts, network_GSGD, yts, NFC, pocket, n_outputs)

                SGDdoTerminate, sgdSR, sgdE = validateSGD(
                    xts, network_SGD, yts, n_outputs)

                plotESGD = np.append(plotESGD, eSGD)
                plotEgensSGD = np.append(plotEgensSGD, t)
                plotEoutSRsgd = np.append(plotEoutSRsgd, sgdSR)

                SGDEout = np.append(SGDEout, sgdE)
                SGDEin = np.append(SGDEin, eSGD)

                if doTerminate:
                    break

                plotEin = np.append(plotEin, ve)
                plotEout = np.append(plotEout, E)
                PlotEoutSR = np.append(PlotEoutSR, SR)
                print("Epoch-"+str(ep+1)+" iteration: "+ str(t))
                print("Success rate of SGD: "+ str(sgdSR))
                print("Success rate of GSGD: "+ str(SR)+ "\n")
        SRovertime.append(SR)
        SRSGDOvertime.append(sgdSR)
        GSGDError.append(E)
        SGDError.append(sgdE)
        best_E= math.inf # draft code
    #write plotting code here
    if(bPlot):

        plt.figure()
        plt.plot(plotEgens, plotE, label='ve', linewidth=1)
        plt.plot(plotEgens, plotEout, 'r--', label='E', linewidth=1)
        plt.plot(plotEgens, PlotEoutSR, 'y-.', label='SR', linewidth=1)

        plt.title('Performance of GSGD - %s' % filePath)
        plt.xlabel("Selected Iterations")
        plt.ylabel("Error (E_i_n/E_v)")
        plt.legend(loc=2)

        plt.show()
        
    if(bPlot):  #temp
        Epocperm = [ i for i in range(n_epoch+1) if i!=0]
        plt.figure()
        plt.plot(Epocperm, SGDError, label='SGD Error', linewidth=1)
        plt.plot(Epocperm, GSGDError, 'r--', label='GSGD Error', linewidth=1)

        plt.title('Error Convergence of GSGD and SGD')
        plt.xlabel("Epochs")
        plt.ylabel("Error")
        plt.legend(loc=2)

        plt.show()
    SR, NFC = PrintFinalResults_updated([], pocket, xts, yts, True, n_outputs)
    #SR, NFC = PrintFinalResults([], pocket, xts, yts, True)
    #sgdSR = PrintFinalResultsSGD(network_SGD, xts, yts)

    print('GSGD: ', SR)
    print('SGD: ', sgdSR)
    #auditing code
    # print("\nSRovertime\n")
    # print(SRovertime)
    # print("SRSGDOvertime\n")
    # print(SRSGDOvertime)
    
    #auditing code
    # print("\nError Over Epochs\n")
    # print("\nGSGD Error over Epoch\n")
    # print(GSGDError)
    # print("\nSGD Error over Epoch\n")
    # print(SGDError)
    return SR, sgdSR
    
def initialize_network(n_hidden, n_inputs , n_outputs):
    this_network = list()
    hidden_layer_matrix_1 = np.random.rand(n_inputs + 1, n_hidden)
    this_network.append(hidden_layer_matrix_1)
    #previuos layer output new layers input
    rows , columns = hidden_layer_matrix_1.shape
    # hidden_layer_matrix_2 = np.random.rand(columns + 1, n_hidden)
    # this_network.append(hidden_layer_matrix_2)
    output_layer = np.random.rand(n_hidden + 1, n_outputs)
    this_network.append(output_layer)
    
    return this_network

def train_network(network, x, y, l_rate, n_outputs, lamda, n_inputs):
    for row, row_label in zip(x,y):
        the_unactivateds, the_activateds = forward_propagate(network, row)
        expected = [0 for i in range(n_outputs)]
        expected[row_label[0]] = 1
        the_deltas = backward_propagate_error(network, np.array(expected), the_activateds)
        update_weights(network, row, l_rate, the_deltas, the_activateds, lamda, n_inputs) 

if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        initialdir=os.path.dirname(os.path.realpath(__file__))+'/data', filetypes=[('data files', '.data')])
    print(file_path)
    if(file_path == ''):
        print('File not found')
    GSGD_ANN(file_path)
