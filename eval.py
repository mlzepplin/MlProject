import scipy.io as sio
import numpy as np

def Error(Ig, Is):
    if len(Ig) != len(Is):
        raise ValueError('the dimesions of segment image and ground image are not same')
    if len(Ig[0]) != len(Is[0]):
        raise ValueError('the dimesions of segment image and ground image are not same')
    intersectionCountMap = {}
    ClusterPixelCountMapForIg = {}
    clusterPixelCountMapForIs = {}

    for row in range(0, len(Is)):  # M
        for col in range(0, len(Is[row])):  # N
            if (Is[row][col] != 0):
                if not Is[row][col] in clusterPixelCountMapForIs:
                    clusterPixelCountMapForIs[Is[row][col]] = 1
                else:
                    clusterPixelCountMapForIs[Is[row][col]] += 1
            
                if not Ig[row][col] in ClusterPixelCountMapForIg:
                    ClusterPixelCountMapForIg[Ig[row][col]] = 1
                else:
                    ClusterPixelCountMapForIg[Ig[row][col]] += 1
                
                if not (Ig[row][col],Is[row][col]) in intersectionCountMap:
                    intersectionCountMap[(Ig[row][col], Is[row][col])] = 1
                else:
                    intersectionCountMap[(Ig[row][col],Is[row][col])] += 1


    totalPixelInGroundTruth = sum(ClusterPixelCountMapForIg.values()) + 0.0
    #print "Total Pixels: "+ str(totalPixelInGroundTruth)

    ErrorGS = 0.0
    for j in ClusterPixelCountMapForIg:
        Wj = ClusterPixelCountMapForIg[j] / totalPixelInGroundTruth
        #print "Wj for Cluster " + str(j) + " is " + str(Wj)
        #print Wj
        ErrorGS += Wj
        denomWji = 0.0
        for k in clusterPixelCountMapForIs:
            if (j, k) in intersectionCountMap:
                denomWji += clusterPixelCountMapForIs[k]
        
        for i in clusterPixelCountMapForIs:
            if (j, i) in intersectionCountMap:
                #denomWji can't be zero
                Wji = clusterPixelCountMapForIs[i] / denomWji
                ErrorGSDenom = clusterPixelCountMapForIs[i] + ClusterPixelCountMapForIg[j] - intersectionCountMap[(j,i)]
                #ErrorGSDenom can't be zero
                ErrorGS -= ((Wj * Wji * intersectionCountMap[(j,i)]) / ErrorGSDenom)
    return ErrorGS


def OCE(Ig, Is):
    return min(Error(Ig,Is),Error(Is,Ig))


def MyClustEvalRGB10(CCIm, GroundTruth):
    # assuming format CCIm as [M*N]
    # ground truth M*N
    
    
    return OCE(GroundTruth,CCIm)

def MyClustEvalHyper10(ClusterIM, GroundTruth):
    #need to fix ground truth mask
    GroundTruthMask=sio.loadmat('/home/ani2404/Desktop/mlproject1/PaviaGrTruthMask.mat')
    #print ClusterIM
    ClusterIM = ClusterIM*GroundTruthMask['PaviaGrTruthMask']
    # print ClusterIM
    return OCE(GroundTruth,ClusterIM)


def MyMaritnIndex(ImageType,LabelImage, GroundTruth):
    if (ImageType == 'RGB'):
        #Need to determine if the input is ClusterIM or CCIm, if
        return MyClustEvalRGB10(LabelImage,GroundTruth)
    else:
        return MyClustEvalHyper10(LabelImage,GroundTruth)


def dispatchForEval321(ImageType,ImageSet, seg1_321,seg2_321,seg3_321):
    
    dim = ImageSet.shape
    M = dim[0]
    N = dim[1]
    numImages = dim[2]
    #output
    eval  = np.zeros((numImages,3))
    
    for i in range(0,numImages):
        print('image count 321:')
        print(i)
        eval[i][0] = MyMaritnIndex(ImageType,ImageSet[:,:,i],seg1_321[:,:,i])
        eval[i][1] = MyMaritnIndex(ImageType,ImageSet[:,:,i],seg2_321[:,:,i])
        eval[i][2] = MyMaritnIndex(ImageType,ImageSet[:,:,i],seg3_321[:,:,i])
    
    return eval


def dispatchForEval481(ImageType, ImageSet, seg1_481,seg2_481,seg3_481):
    
    dim = ImageSet.shape
    M = dim[0]
    N = dim[1]
    numImages = dim[2]
    #output
    eval  = np.zeros((numImages,3))
    
    for i in range(0,numImages):
        print('image count 481:')
        print(i)
        eval[i][0] = MyMaritnIndex(ImageType,ImageSet[:,:,i],seg1_481[:,:,i])
        eval[i][1] = MyMaritnIndex(ImageType,ImageSet[:,:,i],seg2_481[:,:,i])
        eval[i][2] = MyMaritnIndex(ImageType,ImageSet[:,:,i],seg3_481[:,:,i])

    return eval


def main():
    kmeans_RGB_321 = np.load('./images/ClusterIm321_MyKmeans_CCIm.npy')
#    print(kmeans_RGB_321[:,:,0].shape)
#    fcm_RGB_321 = np.load('./images/ClusterIm321_FCM_CCIm.npy')
#    som_RGB_321 = np.load('./images/ClusterIm321_MySOM_CCIm.npy')
#    gmm_RGB_321 = np.load('./images/ClusterIm321_GMM_CCIm.npy')
#    spectral_RGB_321 = np.load('./images/ClusterIm321_MySpectral_CCIm.npy')
#
#    kmeans_RGB_481 = np.load('./images/ClusterIm481_MyKmeans_CCIm.npy')
#    fcm_RGB_481 = np.load('./images/ClusterIm481_FCM_CCIm.npy')
#    som_RGB_481 = np.load('./images/ClusterIm481_MySOM_CCIm.npy')
#    gmm_RGB_481 = np.load('./images/ClusterIm481_GMM_CCIm.npy')
#    spectral_RGB_481 = np.load('./images/ClusterIm481_MySpectral_CCIm.npy')

    seg1_321 = np.load('./images/Seg1_321.npy')
    seg2_321 = np.load('./images/Seg2_321.npy')
    seg3_321 = np.load('./images/Seg3_321.npy')
    
#    print(seg1_321[:,:,0].shape)
#    print(seg2_321[:,:,0].shape)
#    print(seg3_321[:,:,0].shape)

    seg1_481 = np.load('./images/Seg1_481.npy')
    seg2_481 = np.load('./images/Seg2_481.npy')
    seg3_481 = np.load('./images/Seg3_481.npy')

    count =1
    np.save('./evals/rgb/kmeans_321_rgb.npy', dispatchForEval321('RGB',kmeans_RGB_321,seg1_321,seg2_321,seg3_321))
    print(count)
    count=count+1
    np.save('./evals/rgb/fcm_321_rgb.npy', dispatchForEval321('RGB',fcm_RGB_321,seg1_321,seg2_321,seg3_321))
    print(count)
    count=count+1
    np.save('./evals/rgb/som_321_rgb.npy', dispatchForEval321('RGB',som_RGB_321,seg1_321,seg2_321,seg3_321))
    print(count)
    count=count+1
    np.save('./evals/rgb/gmm_321_rgb.npy', dispatchForEval321('RGB',gmm_RGB_321,seg1_321,seg2_321,seg3_321))
    print(count)
    count=count+1
    np.save('./evals/rgb/spectral_321_rgb.npy', dispatchForEval321('RGB',spectral_RGB_321,seg1_321,seg2_321,seg3_321))
    print(count)
    count=count+1

    np.save('./evals/rgb/kmeans_481_rgb.npy', dispatchForEval481('RGB',kmeans_RGB_481, seg1_481,seg2_481,seg3_481))
    print(count)
    count=count+1
    np.save('./evals/rgb/fcm_481_rgb.npy', dispatchForEval481('RGB',fcm_RGB_481, seg1_481,seg2_481,seg3_481))
    print(count)
    count=count+1
    np.save('./evals/rgb/som_481_rgb.npy', dispatchForEval481('RGB',som_RGB_481, seg1_481,seg2_481,seg3_481))
    print(count)
    count=count+1
    np.save('./evals/rgb/gmm_481_rgb.npy', dispatchForEval481('RGB',gmm_RGB_481, seg1_481,seg2_481,seg3_481))
    print(count)
    count=count+1
    np.save('./evals/rgb/spectral_481_rgb.npy', dispatchForEval481('RGB',spectral_RGB_481, seg1_481,seg2_481,seg3_481))
    print(count)
    count=count+1

#mat_contents = sio.loadmat('./images/ClusterIm321_FCM_CCIm.npy')
#    ground_truth = mat_contents['PaviaGrTruth']
#    mat_contents = sio.loadmat('PaviaGrTruthMask.mat')
#    label_image = mat_contents['ClusterIm']
#    print MyMaritnIndex('Hyper',label_image,ground_truth)

if __name__ == '__main__':
    main()

