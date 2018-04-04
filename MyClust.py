# you can choose whether display the segmented images or not by comment/uncomment
# the display syntax at 157-159 lines and 162-164 lines

# to use MyClust function, here is an example:

# from MyClust import MyClust
# ClusterIm, CCIm = MyClust(Im, Algorithm='GMM', ImType='RGB',NumClusts=8)


import scipy.io as sio
import numpy as np
from cv2 import medianBlur
from cv2 import bilateralFilter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


from MyFCM import *
from MyKmeans import *
from MySOM import *     
from MySpectral import * 
from MyGMM import *

from collections import Counter
from skimage import measure
from skimage import morphology

def convertToCCIm(I):

    l1= measure.label(I, connectivity = 2)  # remove areas < 400 pixels
    c1 = Counter(l1.flatten()).most_common()
    l3 = morphology.remove_small_objects(l1,min_size=400,connectivity=2)
    c3 = Counter(l3.flatten()).most_common()

    t=0           
    for i in range(l3.shape[0]):
        if i % 2 ==0:
            for j in range(l3.shape[1]):

                if l3[i,j] == 0:
                    if j < 1:
                        leftbound = 0
                        rightbound = 2
                    elif j > (l3.shape[1] - 2):
                        leftbound = l3.shape[1] - 3
                        rightbound = l3.shape[1] - 1
                    else:
                        leftbound=j-1
                        rightbound=j+1
                    if i < 1:
                        upbound = 0
                        lowbound = 2
                    elif i > (l3.shape[0] - 2):
                        upbound = l3.shape[0] - 3
                        lowbound = l3.shape[0] - 1
                    else:
                        upbound=i-1
                        lowbound=i+1
                    
                    l3neib = l3[upbound:(lowbound+1),leftbound:(rightbound+1)]
                    c0 = Counter(l3neib.flatten()).most_common(2)  
                    if c0[0][0] == 0:
                        if len(c0) > 1:
                            l3[i,j] = c0[1][0]
                    else:
                        l3[i,j] = c0[0][0]
        if i % 2 == 1:
            for j in range(l3.shape[1]-1,-1,-1):
                if t != 0  or l3[i,j] != 0:
                    t = 1
                    if l3[i,j] == 0:
                        if j < 1:
                            leftbound = 0
                            rightbound = 2
                        elif j > (l3.shape[1] - 2):
                            leftbound = l3.shape[1] - 3
                            rightbound = l3.shape[1] - 1
                        else:
                            leftbound=j-1
                            rightbound=j+1
                        if i < 1:
                            upbound = 0
                            lowbound = 2
                        elif i > (l3.shape[0] - 2):
                            upbound = l3.shape[0] - 3
                            lowbound = l3.shape[0] - 1
                        else:
                            upbound=i-1
                            lowbound=i+1
                        
                        l3neib = l3[upbound:(lowbound+1),leftbound:(rightbound+1)]
                        c0 = Counter(l3neib.flatten()).most_common(2)
                        if c0[0][0] == 0:
                            if len(c0)>1:
                                l3[i,j] = c0[1][0]
                        else:
                            l3[i,j] = c0[0][0]
    
    for i in range(l3.shape[0]-1,-1,-1):
        
        for j in range(l3.shape[1]-1,-1,-1):
            i=0
            if l3[i,j] == 0:
                if j < 1:
                    leftbound = 0
                    rightbound = 2
                elif j > (l3.shape[1] - 2):
                    leftbound = l3.shape[1] - 3
                    rightbound = l3.shape[1] - 1
                else:
                    leftbound=j-1
                    rightbound=j+1
                if i < 1:
                    upbound = 0
                    lowbound = 2
                elif i > (l3.shape[0] - 2):
                    upbound = l3.shape[0] - 3
                    lowbound = l3.shape[0] - 1
                else:
                    upbound=i-1
                    lowbound=i+1
                
                l3neib = l3[upbound:(lowbound+1),leftbound:(rightbound+1)]
                c0 = Counter(l3neib.flatten()).most_common(2)
                if c0[0][0] == 0:
                    if len(c0)>1: 
                        l3[i,j] = c0[1][0]
                else:
                    l3[i,j] = c0[0][0]
    l4 = measure.label(l3, connectivity=2)
    return l4




def MyClust(Im, Algorithm, ImType, NumClusts):
    r, c = Im.shape[0:2]
    N = r * c
    numFeature = Im.shape[2]
    if NumClusts > N * 0.25:
        NumClusts = N * 0.25
    elif NumClusts <= 1:
        NumClusts = 0.05 * N

    if ImType == 'RGB':
        Im_tempt = bilateralFilter(Im, 80, 150, 150)
        Im = medianBlur(Im_tempt,9)

    
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

    if ImType == 'RGB':
        CCIm = convertToCCIm(ClusterIm)
##        plt.subplot(121),plt.imshow(ClusterIm),plt.title('ClusterIm')
##        plt.subplot(122),plt.imshow(CCIm),plt.title('CCIm')
##        plt.show()
        return ClusterIm, CCIm
    elif ImType == 'Hyper':
##        plt.imshow(ClusterIm)
##        plt.title('ClusterIm')
##        plt.show()
        return ClusterIm



    
