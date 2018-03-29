import numpy as np
from sklearn.cluster import KMeans


def MyKmeans(Im, ImType, NumClusts):
    ClusterIm, CCIm = 0
    if ImType == 'Hyper':
        CCIm = 0
    r, c = Im.shape[0: 2]
    # Number of clusters
    kmeans = KMeans(n_clusters=NumClusts)

    #Flattening array to vector
    reshapedIm = np.reshape(Im, (len(Im[0])*len(Im),3))

    #Kmeans fitting
    kmeans = kmeans.fit(reshapedIm)

    # Getting the cluster labels
    labels = kmeans.predict(reshapedIm)


    ClusterIm = labels.reshape(len(Im),len(Im[0]))
    # Centroid values
    #centroids = kmeans.cluster_centers
    #plt.imshow(labels)
    #plt.show()
    return ClusterIm, CCIm
