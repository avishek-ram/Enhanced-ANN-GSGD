import torch
import torch.nn as nn
import numpy as np

this_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def getErrorMSE(idx, x, y, network, loss_function):
    data_x = x[idx]
    data_y = y[idx]
    network.zero_grad()
    pred_y = network(data_x.to(device = this_device))
    loss = loss_function(pred_y, data_y)
    return loss.item()