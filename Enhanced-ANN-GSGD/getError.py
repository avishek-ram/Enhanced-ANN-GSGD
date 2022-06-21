import numpy as np
import pandas as pd
import math
from main import custom_train


def getError(network,idx, x, y, n_outputs):
    errors = custom_train(network,x[[idx], :], y[[idx], :],n_outputs)
    output_neurons_error = 3