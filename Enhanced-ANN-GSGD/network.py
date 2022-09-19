# Programmer: Avishek Ram
# email: avishekram30@gmail.com
import numpy as np
import pandas as pd
import math
import os
import copy
from math import exp
import torch
import torch.nn as nn

def train_network(network, x, y, l_rate, n_outputs, lamda, n_inputs, loss_function, optimizer):
    for row, row_label in zip(x,y):
        network.zero_grad()
        data_x = torch.from_numpy(row).float()
        pred_y = network(data_x)
        data_y = torch.from_numpy(row_label).float() #row_label should numpy array
        loss = loss_function(pred_y, data_y)
        loss.backward()

        optimizer.step()