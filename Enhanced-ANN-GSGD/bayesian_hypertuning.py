#Reference - https://towardsdatascience.com/quick-tutorial-using-bayesian-optimization-to-tune-your-hyperparameters-in-pytorch-e9f74fc133c2

from operator import ne
from typing_extensions import Self
import torch
import numpy as np
import torchvision
from torch.utils.data import DataLoader, TensorDataset
from ax.service.managed_loop import optimize
from ax.utils.tutorials.cnn_utils import  train, evaluate

from readData import readData
import tkinter as tk
from tkinter import filedialog
import os
import torch
import torch.nn as nn
from network import *

torch.manual_seed(12345)
#torch.cuda.set_device(0) #this is sometimes necessary for me
dtype = torch.float
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_evaluate(parameterization):
    NC, x, y, N, d, xts, yts = readData('C:/Users/avishek.ram/Documents/GitHub/Enhanced-ANN-GSGD/Enhanced-ANN-GSGD/data/diabetes_readmission_2class.data')

    #training loader
    my_x = x[:1000,:]
    my_y = y[:1000,:]

    tensor_x = torch.Tensor(my_x)
    tensor_y = torch.Tensor(my_y)

    my_dataset = TensorDataset(tensor_x, tensor_y)
    training_loader = DataLoader(my_dataset)

    #testing loader
    test_x = xts
    test_y = yts

    tensor_test_x = torch.Tensor(test_x)
    tensor_test_y = torch.Tensor(test_y)

    test_dataset = TensorDataset(tensor_test_x, tensor_test_y)
    test_loader = DataLoader(test_dataset)
    # constructing a new training data loader allows us to tune the batch size
    # train_loader = torch.utils.data.DataLoader(trainset,
    #                             batch_size=parameterization.get("batchsize", 32),
    #                             shuffle=True,
    #                             num_workers=0,
    #                             pin_memory=True)

    train_loader  = training_loader # or use predefined

    # Get neural net
    untrained_net = init_net(parameterization, d=d) 
    
    # train
    trained_net = net_train(net=untrained_net, train_loader=train_loader, 
                            parameters=parameterization, dtype=dtype, device=device)
        
    # return the accuracy of the model as it was trained in this run
    return evaluate(
        net=trained_net,
        data_loader=test_loader,
        dtype=dtype,
        device=device,
    )

def main_func():
    best_parameters, values, experiment, model = optimize(
        parameters=[
            {"name": "lr", "type": "range", "bounds": [0.0002114354244, 0.00024], "log_scale": True},
            {"name": "lambda", "type": "range", "bounds":[1e-6, 0.9]},
            {"name": "n_hidden", "type": "range", "bounds": [1, 50]},
            #{"name": "stepsize", "type": "range", "bounds": [20, 40]},        
        ],
    
        evaluation_function=train_evaluate,
        objective_name='accuracy',
    )

    print(best_parameters)
    means, covariances = values
    print(means)
    print(covariances)

def net_train(net, train_loader, parameters, dtype, device):

    l_rate = parameters.get("lr", 0.00085)
    lamda = parameters.get("lambda", 0.001)  #Lambda will be used for L2 regularizaion
    betas = (0.9, 0.999)
    beta = 0.9
    epsilon = 1e-8

    net.to(dtype=dtype, device=device)
    criterion = nn.MSELoss()

    optim_params = l_rate, lamda, betas, beta, epsilon
    optims = ['SGD', 'ADAM', 'ADADELTA', 'RMSPROP', 'ADAGRAD']
    
    optim_name = optims[0] #update this to perform hyperparametr tuning for different optimizers
    
    optimizer = get_optimizer(net, name=optim_name, cache= optim_params)

    for _ in range(30):
        for inputs, labels in train_loader:
            # move data to proper dtype and device
            inputs = inputs.to(dtype=dtype, device=device)
            labels = labels.to(device=device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return net

def init_net(parameterization, d):

    #set architure of network here
    model = nn.Sequential(
                      nn.Linear(d, parameterization.get("n_hidden",4)),
                      nn.Sigmoid(),
                      nn.Linear(parameterization.get("n_hidden", 4), 1),
                      nn.Sigmoid())

    return model # return untrained model

if __name__ == '__main__':
    main_func()