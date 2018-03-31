import numpy as np
from skimage import measure
from skimage import filters


def convertToCCIm(I):
    myset = set(I.flatten().tolist())
    maxLabelValue = max(myset)
    C,M,N = len(myset),len(I),len(I[0])
    CCIm = measure.label(I)
    #replacing label 0 pixels with a new maxLabel+1 label
    for m in range(0,M):
        for n in range(0,N):
            if CCIm[m][n] == 0:
                CCIm[m][n] = maxLabelValue+1
    return CCIm
