from __future__ import division, print_function
import numpy as np
import skfuzzy as fuzz
from sklearn import metrics
from sklearn.decomposition import PCA

def MyFCM10(Im, ImageType, k):
    if ImageType == 'Hyper':
        r, c = Im.shape[0:2]
        Im = np.reshape(Im, (r*c, Im.shape[2]))
        pca=PCA(n_components=0.95)
        data = pca.fit_transform(Im)
        data = data.T
    # pre_processing
    if ImageType == 'RGB':
        r, c = Im.shape[0:2]
        data = np.zeros((3, r * c))
        n = -1
        for i in range(r):
            for j in range(c):
                n = n + 1
                data[:, n] = Im[i, j, :]
    metric = []
    for i1 in range(2, k+1):
        _, u, _, _, _, _, fpc = fuzz.cluster.cmeans(data, i1, 2, error=0.005, maxiter=1000, init=None)
        labels = np.argmax(u, axis=0)
        metric.append(metrics.calinski_harabaz_score(data.T, labels))
    metric = np.array(metric)
    index = np.argwhere(metric == max(metric))
    if len(index) > 1:
        index = min(index)
    index = index + 2
    index = int(index)
    _, u, _, _, _, _, _ = fuzz.cluster.cmeans(data, index, 2, error=0.005, maxiter=1000, init=None)
    labels = np.argmax(u, axis=0)
    labels = labels + 1
    ClusterIm = np.zeros((r, c))  
    n2 = -1
    for i2 in range(r):
        for j2 in range(c):
            n2 = n2 + 1
            ClusterIm[i2, j2] = labels[n2]
    return ClusterIm  
