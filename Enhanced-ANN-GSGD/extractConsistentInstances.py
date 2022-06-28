import numpy as np
import pandas as pd
import math
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

def extractConsistentInstances(ve, best_E, omPlusScore, omPlusLevel, omMinuScore, omMinusLevel):

    omMidx = np.nonzero(omMinusLevel > 0)
    omPidx = np.nonzero(omPlusLevel > 0)

    avgMscore = np.sum(omMinusLevel[omMidx])/(np.size(omMidx, 1))
    avgPscoreT = np.sum(omPlusLevel[omPidx])
    avgPscore = avgPscoreT/(np.size(omPidx, 1))  

    omMidx = np.nonzero(omMinusLevel > 0.5*avgMscore)
    omPidx = np.nonzero(omPlusLevel > 0.5*avgPscore)

    if ve < best_E:
        consistentIdx = omMidx[1]
    else:
        consistentIdx = omPidx[1]

    return consistentIdx
