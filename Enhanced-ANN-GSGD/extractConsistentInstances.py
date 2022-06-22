import numpy as np
import pandas as pd
import math
from scipy.special import logsumexp
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

def extractConsistentInstances(ve, best_E, omPlusScore, omPlusLevel, omMinuScore, omMinusLevel):

    #omMidx = find(omMinusLevel>0);
    #omPidx = find(omPlusLevel>0);
    #omMidx = np.nonzero(np.nonzero(omMinusLevel[0]==0))
    omMidx = np.nonzero(omMinusLevel > 0)
    #omPidx = np.nonzero(np.nonzero(omPlusLevel[0]==0))
    omPidx = np.nonzero(omPlusLevel > 0)
    #print("omMidx:123", omMidx)
    #print("omPidx:123:", omPidx)
    # print("omMidxVAL:123::", omMinusLevel[np.where(omMinusLevel>0)]) #returns values that are !=0
    # print("omPidxVAL:123::", omPlusLevel[np.where(omPlusLevel>0)])   #just for checking

    #avgMscore = sum(omMinusLevel(omMidx))/size(omMidx,2);
    #avgPscore = sum(omPlusLevel(omPidx))/size(omPidx,2);
    avgMscore = np.sum(omMinusLevel[omMidx])/(np.size(omMidx, 1))
    avgPscoreT = np.sum(omPlusLevel[omPidx])
    avgPscore = avgPscoreT/(np.size(omPidx, 1))  # 0=row 1=col
    #print("avgMscore::", avgMscore)
    #print("avgPscore::", avgPscore)

    #omMidx = find(omMinusLevel>0.5*avgMscore);
    #omPidx = find(omPlusLevel>0.5*avgPscore);
    #omMidx = np.nonzero(np.nonzero(omMinusLevel[0]==0.5*avgMscore))
    omMidx = np.nonzero(omMinusLevel > 0.5*avgMscore)
    #omPidx = np.nonzero(np.nonzero(omPlusLevel[0]==0.5*avgPscore))
    omPidx = np.nonzero(omPlusLevel > 0.5*avgPscore)
    # print("omMidx`````````",omMidx[1])
    # print("omPidx`````````",omPidx[1])
    # print("omMidxVAL:::", omMinusLevel[np.where(omMinusLevel>0)]) #returns values that are !=0
    #print("omPidxVAL:::", omPlusLevel[np.where(omPlusLevel>0)])
    # print("VE______",ve,"____best_E_____",best_E)
    #print("omPlusScore::", omPlusScore)
    #print("omPlusLevel::", omPlusLevel)
    #print("omMinuScore::", omMinuScore)
    #print("omMinusLevel::", omMinusLevel)

    if ve < best_E:
        consistentIdx = omMidx[1]
        #print("ve < best_E finally ------------------------------------------------------------------------------------------------------------")
    else:
        consistentIdx = omPidx[1]

    return consistentIdx
