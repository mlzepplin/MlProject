import numpy as np


def MySpectral(Im, ImType, NumClusts):
    ClusterIm, CCIm = 0
    if ImType == 'Hyper':
        CCIm = 0
    r, c = Im.shape[0: 2]
    print(r)
    print(c)
    return ClusterIm, CCIm
