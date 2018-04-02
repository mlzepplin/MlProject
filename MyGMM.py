# coding=utf-8
# test

# cluster validity pick one
# objective function except kmenas
# kmeans saperation indices   scatter metrices

# fcm gmm objective function
# som scatter metrices
# spectral 37'20 kmeans?

# kmeans rather than knn for spectral


# 要不要smoothpicture得再看


# GMM
from sklearn.mixture import GaussianMixture
import numpy as np


def MyGMM(Im, ImageType, k):
    # pre-processing
    if ImageType == 'RGB':
        r, c = Im.shape[0:2]
        data = np.zeros((r * c, 3))
        n = -1
        for i in range(r):
            for j in range(c):
                n = n + 1
                data[n, :] = Im[i, j, :]
    # row=sample;column=dimension
    bicscore = np.zeros(k)
    for k0 in range(2, k + 2):
        gmm = GaussianMixture(n_components=k0).fit(data)
        bicscore[k0 - 2] = gmm.bic(data)
    index = np.argwhere(bicscore == min(bicscore))
    if len(index) > 1:
        index = min(index)
    index = index + 2
    index = int(index)
    gmm = GaussianMixture(n_components=(index)).fit(data)
    labels = gmm.predict(data)
    ClusterIm = np.zeros((r, c))  # 注意是float，是否改成uint8
    labels = labels + 1
    n0 = -1
    for i1 in range(r):
        for j1 in range(c):
            n0 = n0 + 1
            ClusterIm[i1, j1] = labels[n0]
    return ClusterIm  # 目前只返回这个 CCIm最后再统一写个函数转换
