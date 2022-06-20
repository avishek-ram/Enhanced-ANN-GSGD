import numpy as np
import pandas as pd
import math
import os
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from math import exp
from random import seed
from random import randrange
from random import random 


def readData(file_path):
    data = pd.read_csv(file_path, header=None, index_col=None)

    data = pd.DataFrame(data)  # creates a dataframe from datafile
    # randomise rows of the data file,
    data = data.iloc[np.random.permutation(len(data.index))]

    N = len(data.index)  # number of rows
    d = len(data.columns)  # number of columns

    training = data.head(math.ceil(N*0.6))  # get the top 60% of rows
    testing = data.tail(math.floor(N*0.4))  # get the bottom 40% of rows

    getRows = len(training.index)  # get number of rows in training data
    getColumn = len(training.columns)  # get number of columns in training data

    y = training[[d-1]]  # saving the class variable to y
    # saving the input variables to x (does not contain the leading zeros)
    x = training.iloc[:, :-1]

    x = (x-x.min())/(x.max()-x.min())  # Normalising data
    x = x.reset_index(drop=True)  # resets the indexes
    # x1s = pd.DataFrame(np.ones((getRows, 1), dtype=int), columns=[
    #                    'ones'])  # create a column of ones
    # x = x1s.join(x)  # adding ones in the beginning

    N = len(x.index)  # update N with training data
    d = len(x.columns)  # update d with training data
    indices = np.arange(d)  # 0,1,2...(d-1)

    x = np.array(x.values.tolist())
    y = np.array(y.values.tolist())

    NC = np.max(np.size(np.unique(y)))  # get maximum class value count,

    # testing data
    getRowsts = len(testing.index)  # get number of rows in training data
    # get number of columns in training data
    getColumnts = len(testing.columns)
    yts = testing[[d]]  # saving the class variable to y [:,:-1]
    # testing[[0,1,2,3,4,5,6,7,8]] #saving the input variables to x (does not contain the leading ones)
    xts = testing.iloc[:, :-1]
    xts = (xts-xts.min())/(xts.max()-xts.min())  # Normalising data
    xts = xts.reset_index(drop=True)  # resets the indexes

    # x1sts = pd.DataFrame(np.ones((getRowsts, 1), dtype=int), columns=[
    #                      'ones'])  # create a column of ones
    # xts = x1sts.join(xts)  # adding ones in the beginning

    Nts = len(xts.index)  # update N with training data
    dts = len(xts.columns)  # update d with training data

    xts = np.array(xts.values.tolist())
    yts = np.array(yts.values.tolist())

    return NC, x, y, N, d, xts, yts