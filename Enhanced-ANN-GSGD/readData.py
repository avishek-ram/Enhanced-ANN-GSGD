import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import train_test_split


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
        #avishek- start oversampling minority class in training data only
        
        #print currnt classification class distribution
        #counter = Counter(tuple(te) for te in Y_train_old.values)
        #print(counter)
        
        oversample = SMOTE(random_state=20)
    
        x, y = oversample.fit_resample(X_train_old,Y_train_old)
        
        #print classification class distribution
        # counter = Counter(tuple(te) for te in y.values)
        # print(counter)
        #end
    else:
        x = X_train_old
        y = Y_train_old
    
    x = (x-x.min())/(x.max()-x.min())  # Normalising data
    x = x.reset_index(drop=True)  # resets the indexes
    # x1s = pd.DataFrame(np.ones((getRows, 1), dtype=int), columns=[
    #                    'ones'])  # create a column of ones
    # x = x1s.join(x)  # adding ones in the beginning

    N = len(x.index)  # update N with training data
    d = len(x.columns)  # update d with training data

    x = np.array(x.values.tolist())
    y = np.array(y.values.tolist())
    
    NC = np.max(np.size(np.unique(y)))  # get maximum class value count,

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