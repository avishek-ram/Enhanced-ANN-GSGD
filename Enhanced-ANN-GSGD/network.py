# Programmer: Avishek Ram
# email: avishekram30@gmail.com
import numpy as np
from math import exp
import torch
import torch.nn as nn

this_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_network(network, data_x, data_y, loss_function, optimizer):
    network.zero_grad()
    pred_y = network(data_x)
    loss = loss_function(pred_y, data_y)
    loss.backward()
    optimizer.step()

def get_optimizer(network, name, cache):
    l_rate, lamda, betas, beta, epsilon = cache 
    if(name == 'SGD'):
        return torch.optim.SGD(network.parameters(), lr=l_rate, weight_decay= lamda, momentum=0.1)
        #return torch.optim.SGD(network.parameters(), lr=l_rate, weight_decay= lamda)
    elif(name == 'ADAM'):
        return torch.optim.Adam(network.parameters(), lr=l_rate, betas= betas,  weight_decay= lamda)
    elif(name == 'ADADELTA'):
        return torch.optim.Adadelta(network.parameters(), lr=l_rate, eps= epsilon,rho= beta, weight_decay = lamda)
    elif(name == 'RMSPROP'):
        return torch.optim.RMSprop(network.parameters(), lr=l_rate, eps= epsilon, weight_decay = lamda)
    elif(name == 'ADAGRAD'):
        return torch.optim.Adagrad(network.parameters(), lr=l_rate, eps= epsilon, weight_decay = lamda)