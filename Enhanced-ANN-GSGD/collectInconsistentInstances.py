from getError import getError
import numpy as np
import pandas as pd
import math


def collectInconsistentInstances(idx, x, y, network_GSGD, ropeTeamSz, pe, n_outputs):

    omPlusScore = np.zeros((1, ropeTeamSz))
    omPlusLevel = np.zeros((1, ropeTeamSz))
    omMinuScore = np.zeros((1, ropeTeamSz))
    omMinusLevel = np.zeros((1, ropeTeamSz))

    tmpGuided = 0
    # print('idx',idx.size)

    for j in range(0, min(ropeTeamSz, idx.size)):
        tmpGuided = tmpGuided+1
        nErr = getError(idx[j], x, y, network_GSGD, n_outputs)#idx[j], x, y, w)
        # print('------------------------------')
        #print('nErr: ', nErr)
        #print('pe: ', pe)
        if nErr > pe:
            omPlusScore[:, j] = omPlusScore[:, j] + 1
            omPlusLevel[:, j] = omPlusLevel[:, j] + (nErr - pe)

        else:
            omMinuScore[:, j] = omMinuScore[:, j] + 1
            omMinusLevel[:, j] = omMinusLevel[:, j] + (pe - nErr)

    return omPlusScore, omPlusLevel, omMinuScore, omMinusLevel, tmpGuided
