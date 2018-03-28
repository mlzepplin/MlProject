import scipy.io as sio
import numpy as np

from MyFCM import MyFCM
from MyKmeans import MyKmeans
from MySOM import MySOM
from MySpectral import MySpectral
from MyGMM import MyGMM


def MyClust(Im, Algorithm, ImType, NumClusts):
    N = Im.size / 3
    r, c = Im.shape[0:2]
    if NumClusts > N * 0.25:
        NumClusts = N * 0.25
    elif NumClusts <= 1:
        NumClusts = 0.05 * N

    if Algorithm == 'Kmeans':
        ClusterIm = MyKmeans(Im, ImType, NumClusts)
    elif Algorithm == 'SOM':
        ClusterIm = MySOM(Im, ImType, NumClusts)
    elif Algorithm == 'FCM':
        ClusterIm = MyFCM(Im, ImType, NumClusts)
    elif Algorithm == 'Spectral':
        ClusterIm = MySpectral(Im, ImType, NumClusts)
    elif Algorithm == 'GMM':
        ClusterIm = MyGMM(Im, ImType, NumClusts)
    else:
        print("Wrong Algorithm Input")

    if ImType == 'RGB':
        CCIm = np.zeros((r, c))
        return ClusterIm, CCIm
    elif ImType == 'Hyper':
        return ClusterIm


if __name__ == '__main__':
    matfn = 'ImsAndTruths2092.mat'
    data = sio.loadmat(matfn)
    Im = data['Im']
    Algorithm = 'FCM'  # raw_input("Algorithm = ")
    ImType = 'RGB'  # raw_input("ImType = ")
    NumClusts = 4  # int(input("NumClusts = "))
    ClusterIm, CCIm = MyClust(Im, Algorithm, ImType, NumClusts)
