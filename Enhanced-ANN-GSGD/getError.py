import torch
import torch.nn as nn
import numpy as np

def getErrorMSE(idx, x, y, network, loss_function):
    row = x[idx]
    row_label = y[idx]
    network.zero_grad()
    data_x = torch.from_numpy(row).float()
    pred_y = network(data_x)
    data_y = torch.from_numpy(row_label).float() #row_label should numpy array
    loss = loss_function(pred_y, data_y)
    return loss.item()