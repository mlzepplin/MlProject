from sklearn import metrics
from sklearn.mixture import GaussianMixture
import numpy as np
from sklearn.decomposition import PCA

def MyGMM(Im, ImageType, k):
    if ImageType == 'Hyper':
        r, c = Im.shape[0:2]
        Im = np.reshape(Im, (r*c, Im.shape[2]))
        pca=PCA(n_components=0.95)
        data = pca.fit_transform(Im)   
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
    bicscore = np.zeros(k-1)
    for k0 in range(2,k+1):
        gmm = GaussianMixture(n_components=k0).fit(data)
        bicscore[k0 - 2] = gmm.bic(data)
    index = np.argwhere(bicscore == min(bicscore))
    if len(index) > 1:
        index = min(index)
    index = index + 2
    index = int(index)
    gmm = GaussianMixture(n_components=(index)).fit(data)
    labels = gmm.predict(data)
    ClusterIm = np.zeros((r, c))  
    labels = labels + 1
    n0 = -1
    for i1 in range(r):
        for j1 in range(c):
            n0 = n0 + 1
            ClusterIm[i1, j1] = labels[n0]
    return ClusterIm  

