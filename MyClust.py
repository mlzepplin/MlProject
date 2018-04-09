# you can choose whether display the segmented images or not by comment/uncomment
# the display syntax at 157-159 lines and 162-164 lines

# to use MyClust function, here is an example:

# from MyClust import MyClust
# ClusterIm, CCIm = MyClust(Im, Algorithm='GMM', ImType='RGB',NumClusts=8)

from cv2 import medianBlur
from cv2 import bilateralFilter



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




def MyClust10(Im, Algorithm, ImType, NumClusts):

    #Type checking for safety
    if(isinstance(Algorithm,str) == False):
        raise ValueError( "Algorithm should be of type string. Please provide either 'Kmeans' or 'SOM' or 'FCM' or 'Spectral' or 'GMM'")
    elif (isinstance(ImType,str) == False):
        raise ValueError("ImType should be of type string. Please provide either 'RGB' or 'Hyper'")
    elif (isinstance(Im, np.ndarray) == False):
        raise ValueError("Im should be a numpy array. Please provide numpy array of dimension(r,c,featureSize)")
    elif len(Im.shape) != 3:
        raise ValueError("Im shape is invalid. Please provide numpy array of shape (rows,col,featureSize)")

    r, c = Im.shape[0:2]
    N = r * c
    numFeature = Im.shape[2]
    if NumClusts > N * 0.25:
        NumClusts = N * 0.25
    elif NumClusts <= 1:
        NumClusts = 0.05 * N

    if ImType == 'RGB':
        Im = bilateralFilter(Im, 80, 150, 150)
        Im = medianBlur(Im,9)
    elif ImType != 'Hyper':
        raise ValueError(
            "Invalid ImType :" + ImType + ". Please provide either 'RGB' or 'Hyper'")

    
    if Algorithm == 'Kmeans':
        ClusterIm = MyKmeans10(Im, ImType, NumClusts)
    elif Algorithm == 'SOM':
        ClusterIm = MySOM10(Im, ImType, NumClusts)
    elif Algorithm == 'FCM':
        ClusterIm = MyFCM10(Im, ImType, NumClusts)
    elif Algorithm == 'Spectral':
        ClusterIm = MySpectral10(Im, ImType, NumClusts)
    elif Algorithm == 'GMM':
        ClusterIm = MyGMM10(Im, ImType, NumClusts)
    else:
        raise ValueError("Invalid Algorithm name:"+Algorithm +". Please provide either 'Kmeans' or 'SOM' or 'FCM' or 'Spectral' or 'GMM'")


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



    
