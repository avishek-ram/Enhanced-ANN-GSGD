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

def GSGD_ANN(filePath):
    # reading data, normalize and spliting into train/test
    NC, x, y, N, d, xts, yts = readData(filePath)

    # Test Backprop on Seeds dataset
    seed(1)
    # evaluate algorithm
    n_folds = 10
    l_rate = 0.3
    n_epoch = 500
    n_hidden = 5
    
    scores = evaluate_algorithm(back_propagation, x, y, xts, yts , l_rate, n_hidden, d, NC)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
    
    #have to show some kind od score here and mean accuracy
    
def evaluate_algorithm(algorithm, x, y, xts, yts , l_rate, n_hidden, d, NC):
    #scores #have to return this
    scores = list()
    predicted = algorithm(x, y, xts, yts, l_rate, n_hidden, d, NC)
    actual = yts
    accuracy = accuracy_metric(actual, predicted)
    scores.append(accuracy)
    
    return scores
    
def back_propagation(x, y, xts, yts, l_rate, n_hidden, n_inputs, n_outputs):
    network = initialize_network(n_hidden, n_inputs , n_outputs)
    for t in range(500):
        temp = np.append(x,y,axis=1)
        np.random.shuffle(temp)
        x = temp[:,:-1] 
        y = np.array(temp[:,-1], dtype= np.int64).reshape(len(y),1)
        train_network(network, x, y, l_rate, n_outputs)
    
    predictions_test = list()
    for row in xts:
        prediction = predict(network, row)
        predictions_test.append([prediction])
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
        
# Make a prediction with a network
def predict(network, row):
	unactivated_outputs, activated_output = forward_propagate(network, row)
	return np.argmax(activated_output[-1])

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i][0] == predicted[i][0]:
			correct += 1
	return correct / float(len(actual)) * 100.0

if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        initialdir=os.path.dirname(os.path.realpath(__file__))+'/data', filetypes=[('data files', '.data')])
    print(file_path)
    if(file_path == ''):
        print('File not found')
    GSGD_ANN(file_path)
