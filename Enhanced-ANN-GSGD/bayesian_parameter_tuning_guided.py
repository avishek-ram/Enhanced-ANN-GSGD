#Reference - https://towardsdatascience.com/quick-tutorial-using-bayesian-optimization-to-tune-your-hyperparameters-in-pytorch-e9f74fc133c2

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from ax.service.managed_loop import optimize
from ax.utils.tutorials.cnn_utils import  train, evaluate # evaluate can also be used ANN

from readData import readData
import torch
import torch.nn as nn
from random import seed


seed(1)

#torch.cuda.set_device(0) #this is sometimes necessary for me
dtype = torch.float
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_evaluate(parameterization):
    NC, x, y, N, d, xts, yts = readData('/home/paperspace/Documents/Enhanced-ANN-GSGD/Enhanced-ANN-GSGD/data/diabetes_readmission_2class.data')

    #training loader
    my_x = x
    my_y = y

    tensor_x = torch.Tensor(my_x)
    tensor_y = torch.Tensor(my_y)

    my_dataset = TensorDataset(tensor_x, tensor_y)
    training_loader = DataLoader(my_dataset, batch_size= parameterization.get("batch_size", 7), shuffle= True)

    #testing loader
    test_x = xts
    test_y = yts

    tensor_test_x = torch.Tensor(test_x)
    tensor_test_y = torch.Tensor(test_y)

    test_dataset = TensorDataset(tensor_test_x, tensor_test_y)
    test_loader = DataLoader(dataset= test_dataset)
    # constructing a new training data loader allows us to tune the batch size
    
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
            {"name": "lr", "type": "range", "bounds": [1e-7, 0.9], "log_scale": True},
            {"name": "lambda", "type": "range", "bounds":[1e-7, 0.9]},
            #{"name": "momentum", "type": "range", "bounds":[1e-20, 1.0]},
            {"name": "n_hiddenA", "type": "range", "bounds": [300, 400]},
            {"name": "batch_size", "type": "range", "bounds": [1, 1000]},        
            #{"name": "dampening", "type": "range", "bounds": [0.0, 0.9]},        
            {"name": "epochs", "type": "range", "bounds": [1, 30]},        
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
    #momentum =  parameters.get("momentum", 0.9) 
    # dampening = parameters.get("dampening","0.9")
    betas = (0.9, 0.999)
    beta = 0.9
    epsilon = 1e-8

    net.to(dtype=dtype, device=device)
    criterion = nn.MSELoss()

    optim_params = l_rate, lamda, betas, beta, epsilon #, momentum, dampening
    optims = ['SGD', 'ADAM', 'ADADELTA', 'RMSPROP', 'ADAGRAD']
    
    optim_name = optims[0] #update this to perform hyperparametr tuning for different optimizers
    
    optimizer = get_optimizer(net, name=optim_name, cache= optim_params)

    for _ in range(parameters.get("epochs",10)):
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
                      nn.Linear(d, parameterization.get("n_hiddenA",4)),
                      nn.Sigmoid(),
                      nn.Linear(parameterization.get("n_hiddenA", 4), 1),
                      nn.Sigmoid()).to(device=device)

    return model # return untrained model

def get_optimizer(network, name, cache):
    l_rate, lamda, betas, beta, epsilon = cache 
    if(name == 'SGD'):
        return torch.optim.SGD(network.parameters(), lr=l_rate, weight_decay= lamda)
    elif(name == 'ADAM'):
        return torch.optim.Adam(network.parameters(), lr=l_rate, betas= betas,  weight_decay= lamda)
    elif(name == 'ADADELTA'):
        return torch.optim.Adadelta(network.parameters(), lr=l_rate, eps= epsilon,rho= beta, weight_decay = lamda)
    elif(name == 'RMSPROP'):
        return torch.optim.RMSprop(network.parameters(), lr=l_rate, eps= epsilon, weight_decay = lamda)
    elif(name == 'ADAGRAD'):
        return torch.optim.Adagrad(network.parameters(), lr=l_rate, eps= epsilon, weight_decay = lamda)

if __name__ == '__main__':
    main_func()