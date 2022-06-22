from readData import readData
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os
from math import exp
from random import seed
from random import randrange
from random import random  # check if seed is used or not
from propagation import *
from getError import getError
from collectInconsistentInstances import collectInconsistentInstances
from extractConsistentInstances import extractConsistentInstances
from validate import *
from printFinalResults import *

def GSGD_ANN(filePath):
    # reading data, normalize and spliting into train/test
    NC, x, y, N, d, xts, yts = readData(filePath)

    # Test Backprop on Seeds dataset
    seed(1)
    # evaluate algorithm
    n_folds = 10
    l_rate = 0.1
    n_epoch = 500
    n_hidden = 5
    
    scores = evaluate_algorithm(back_propagation, x, y, xts, yts , l_rate, n_hidden, d, NC, N)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
    
    #have to show some kind od score here and mean accuracy
    
def evaluate_algorithm(algorithm, x, y, xts, yts , l_rate, n_hidden, d, NC, N):
    #scores #have to return this
    scores = list()
    algorithm(x, y, xts, yts, l_rate, n_hidden, d, NC, N)
    #predicted = algorithm(x, y, xts, yts, l_rate, n_hidden, d, NC, N)
    #actual = yts
    #accuracy = accuracy_metric(actual, predicted)
    #scores.append(accuracy)
    
    return scores
    
def back_propagation(x, y, xts, yts, l_rate, n_hidden, n_inputs, n_outputs, N):
    #below code is just randomizing and retrainng  temporary
    #network = initialize_network(n_hidden, n_inputs , n_outputs)
    # for t in range(500):
    #     temp = np.append(x,y,axis=1)
    #     np.random.shuffle(temp)
    #     x = temp[:,:-1] 
    #     y = np.array(temp[:,-1], dtype= np.int64).reshape(len(y),1)
    #     train_network(network, x, y, l_rate, n_outputs)
    # end
    
    #Start GSGD implemenattion here
    ropeTeamSz = 10  # this is rho. neighborhood size => increase rho value when the dataset is very noisy.
    pe = math.inf
    t = 0
    idx = np.array(np.random.permutation(N-1))
    idxs = idx
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
    SGDpocket = PGW()
    
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

    bPlot = True
    
    #initialize network here / inital weights
    network_GSGD = initialize_network(n_hidden, n_inputs , n_outputs)
    network_SGD = initialize_network(n_hidden, n_inputs , n_outputs)
    
    T = 1000
    for t in range(T):
        et = et + 1
        # if not idx[et]:
        if idx.size == 0:
            idx = np.random.permutation(N-1)
            et = 0
            idxs = idx  
        
        curIdx = idx[et]
        
        # update weights for the iteration
        train_network(network_GSGD, x[[curIdx],:], y[[curIdx],:], l_rate, n_outputs)
        NFC = NFC + 1
        #end
        
        er = np.random.permutation(N)
        er = np.array([er])
                
        #get initial SGD weights
        curIdxs = idxs[0]
        idxs = np.delete(idxs, 0, axis=0)
        train_network(network_SGD, x[[curIdxs],:], y[[curIdxs],:], l_rate, n_outputs)
        SGDnfc = SGDnfc + 1
        #end
        
        ve = 0
        eSGD = 0
        
        #with the initialized weight we will now try to get averaged error value of a few random rows for both
        for k in er[0]:
            ve = ve + getError(k, x, y, network_GSGD, n_outputs)
            eSGD = eSGD + getError(k, x, y, network_SGD, n_outputs)
        
        ve = ve/len(er[0])
        eSGD = eSGD/len(er[0])
        
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
            train_network(network_GSGD, x[[curId],:], y[[curId],:], l_rate, n_outputs)
            NFC = NFC+1
            
        if t % ropeTeamSz == 0 or t % tmpGuided == 0:  # or et > idx.size:
            # print("heeerrre before idx.size ",idx.size)
            idx = np.delete(idx, np.s_[0:tmpGuided], axis=0)
            et = -1
        
        else:
            # tmpVals = np.setdiff1d(idx[0:tmpGuided], idx[consistentIdx])
            tmpVals = np.setdiff1d(idx[0:tmpGuided], idx[consistentIdx])
            idxArray = np.arange(tmpGuided)
            inconsistentIdx = np.setdiff1d(idxArray, consistentIdx)

            # remove consistent idx from list to only have inconsistent idx... 0 is an index** CHECK THIS
            idx = np.delete(idx, inconsistentIdx, axis=0)
            #print("idx.size ",idx.size) #np.array(0)#

            idx = np.r_[idx, tmpVals]
            # idx = np.append(idx, tmpVals)
            # idx = np.hstack((idx, tmpVals))
        
        # Plot section
        if t % 10 == 0 or t == T:

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
    
    #write plotting code here

    SR, NFC = PrintFinalResults([], pocket, xts, yts, True)
    sgdSR = PrintFinalResultsSGD(network_SGD, xts, yts)

    print('GSGD: ', SR)
    print('SGD: ', sgdSR)

    return SR, sgdSR
    
    predictions_test = list()
    # for row in xts:
    #     prediction = predict(network, row)
    #     predictions_test.append([prediction])
    return (predictions_test)
    
def initialize_network(n_hidden, n_inputs , n_outputs):
    this_network = list()
    hidden_layer_matrix_1 = np.random.rand(n_inputs + 1, n_hidden)
    this_network.append(hidden_layer_matrix_1)
    #previuos layer output new layers input
    rows , columns = hidden_layer_matrix_1.shape
    hidden_layer_matrix_2 = np.random.rand(columns + 1, n_hidden)
    this_network.append(hidden_layer_matrix_2)
    output_layer = np.random.rand(n_hidden + 1, n_outputs)
    this_network.append(output_layer)
    
    return this_network

def train_network(network, x, y, l_rate, n_outputs):
    hot_encoded_labels = []
    for row, row_label in zip(x,y):
        the_unactivateds, the_activateds = forward_propagate(network, row)
        expected = [0 for i in range(n_outputs)]
        expected[row_label[0]] = 1
        the_deltas =backward_propagate_error(network, np.array(expected), the_activateds)
        update_weights(network, row, l_rate, the_deltas, the_activateds) 
        #print(x)

if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        initialdir=os.path.dirname(os.path.realpath(__file__))+'/data', filetypes=[('data files', '.data')])
    print(file_path)
    if(file_path == ''):
        print('File not found')
    GSGD_ANN(file_path)
