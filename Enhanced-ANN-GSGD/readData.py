import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def readData(file_path):
    data = pd.read_csv(file_path, header=None, index_col=None)

    data = pd.DataFrame(data)  # creates a dataframe from datafile
    # randomise rows of the data file,
    data = data.iloc[np.random.permutation(len(data.index))]

    N = len(data.index)  # number of rows
    d = len(data.columns)  # number of columns
    
    X_train_old, xts, Y_train_old, yts = train_test_split(data.iloc[:, :-1], data[[d-1]] , test_size=0.20, random_state=0)
    
    apply_smote_oversampling = True
    
    if apply_smote_oversampling:
        #Avishek- start oversampling minority class in training data only    
        oversample = SMOTE()
        x, y = oversample.fit_resample(X_train_old,Y_train_old)
    else:
        x = X_train_old
        y = Y_train_old

    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x.values)
    x = pd.DataFrame(x_scaled)

    N = len(x.index)  # update N with training data
    d = len(x.columns)  # update d with training data

    x = np.array(x.values.tolist())
    y = np.array(y.values.tolist())
    
    NC = np.max(np.size(np.unique(y)))  # get maximum class value count,

    xts_scaled = min_max_scaler.fit_transform(xts.values)
    xts = pd.DataFrame(xts_scaled)

    xts = np.array(xts.values.tolist())
    yts = np.array(yts.values.tolist())

    return NC, x, y, N, d, xts, yts