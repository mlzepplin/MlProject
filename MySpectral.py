import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn import metrics
import cv2
from sklearn.decomposition import PCA

def MySpectral(Im, ImType, k):

    if ImType == 'Hyper':
        r, c = Im.shape[0:2]
        Im = np.reshape(Im, (r*c, Im.shape[2]))
        pca=PCA(n_components=4)
        Im_4d = pca.fit_transform(Im)
        n7 = -1
        Im = np.zeros((r,c,4))
        for i5 in range(r):
            for j6 in range(c):
                n7 = n7 + 1
                Im[i5, j6, :] = Im_4d[n7,:]
        
    # preprocessing downsampling
    r_raw, c_raw = Im.shape[0:2]
    if ImType == 'Hyper':
        if min(r_raw,c_raw)<250:
            dsize_row = Im.shape[0]//2
            dsize_col = Im.shape[1]//2
        else:
            dsize_row = Im.shape[0]//3
            dsize_col = Im.shape[1]//3           
    if ImType == 'RGB':
        dsize_row = Im.shape[0]//5
        dsize_col = Im.shape[1]//5   
    Im = cv2.resize(src=Im,dsize=(dsize_col,dsize_row),interpolation=cv2.INTER_AREA)

    if ImType == 'Hyper':
        r, c = Im.shape[0:2]
        data = np.reshape(Im, (r*c, Im.shape[2]))
        
    if ImType == 'RGB':
        r, c = Im.shape[0:2]
        data = np.zeros((r * c, 3))
        n = -1
        for i in range(r):
            for j in range(c):
                n = n + 1
                data[n, :] = Im[i, j, :]
        
    metric = np.zeros(k-1)
    for i0 in range(2, k + 1):

        # if memory problem, divide data
        # if too slow, then eigen_solver could set to 'amg' for large sparse problems
        # if too fast ,then affinity could be rbf
        # n_neighbors should be decided by the size of affinity matrix
        y_predict = SpectralClustering(n_clusters=i0,affinity='nearest_neighbors',
                                n_neighbors=5,n_jobs=-1).fit_predict(data)
        metric[i0-2] = metrics.calinski_harabaz_score(data, y_predict)

    index = np.argwhere(metric == max(metric))
    if len(index) > 1:
        index = min(index)
    index = index + 2
    index = int(index)
    labels = SpectralClustering(n_clusters=i0,affinity='nearest_neighbors',
                                n_neighbors=5,n_jobs=-1).fit_predict(data)

    labels = labels + 1
    ClusterIm = np.zeros((r,c))
    n2 = -1
    for i2 in range(r):
        for j2 in range(c):
            n2 = n2 + 1
            ClusterIm[i2, j2] = labels[n2]
    #postpro upsampling
    ClusterIm = cv2.resize(src=ClusterIm,dsize=(c_raw,r_raw),interpolation=cv2.INTER_NEAREST)      
    return ClusterIm
      
