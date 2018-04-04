import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.decomposition import PCA

def MyKmeans(Im, ImageType, k):

    if ImageType == 'Hyper':
        r, c = Im.shape[0:2]
        Im = np.reshape(Im, (r*c, Im.shape[2]))
        pca=PCA(n_components=0.95)
        data = pca.fit_transform(Im)   
    # pre_processing
    if ImageType == 'RGB':
        r, c = Im.shape[0:2]
        data = np.zeros((r * c, 3))
        #    r,c,p = self._weights.shape[0:3]
        #    weights = np.zeros((r*c,p))
        n = -1
        for i in range(r):
            for j in range(c):
                n = n + 1
                data[n, :] = Im[i, j, :]

    metric = []
    for i1 in range(2,k+1):
        kmeans = KMeans(n_clusters=i1)
        kmeans.fit(data)
        labels = kmeans.labels_
        metric.append(metrics.calinski_harabaz_score(data, labels))
    metric = np.array(metric)
    index = np.argwhere(metric == max(metric))
    index = index + 2
    if len(index) > 1:
        index = min(index)
    index = int(index)
    kmeans1 = KMeans(n_clusters=index)
    kmeans1.fit(data)
    labels1 = kmeans1.labels_ + 1
    ClusterIm = np.zeros((r, c))  
    n2 = -1
    for i2 in range(r):
        for j2 in range(c):
            n2 = n2 + 1
            ClusterIm[i2, j2] = labels1[n2]
    return ClusterIm  


