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
    #need to fix ground truth mask multiplication
    #TODO, we don't know if the input will already be multiplied
    # print ClusterIM
    return OCE(GroundTruth,ClusterIM)


def MyMaritnIndex10(ImageType,LabelImage, GroundTruth):
    #Type checking for safety
    if (isinstance(ImageType,str) == False):
        raise ValueError("ImType should be of type string. Please provide either 'RGB' or 'Hyper'")
    elif (isinstance(LabelImage, np.ndarray) == False):
        raise ValueError("LabelImage should be a numpy array. Please provide numpy array of dimension(r,c,featureSize)")
    elif (isinstance(GroundTruth, np.ndarray) == False):
        raise ValueError("GroundTruth should be a numpy array. Please provide numpy array of dimension(r,c,featureSize)")

    if (ImageType == 'RGB'):
        #Need to determine if the input is ClusterIM or CCIm, if
        return MyClustEvalRGB10(LabelImage,GroundTruth)
    elif (ImageType == 'Hyper'):
        return MyClustEvalHyper10(LabelImage,GroundTruth)
    else:
        raise ValueError("Wrong imageType entered. Please provide either 'RGB' or 'Hyper'")

############################################
#FOR TESTING PURPOSES ON OUR 198 IMAGES ONLY
############################################
def dispatchForEval(ImageType, ImageSet, seg1,seg2,seg3):
    #seg1,2,3 are the human marked groundtruths
    dim = ImageSet.shape
    M = dim[0]
    N = dim[1]
    numImages = dim[2]
    #output
    eval  = np.zeros((numImages,3))
    
    for i in range(0,numImages):
        eval[i][0] = MyMaritnIndex10(ImageType,ImageSet[:,:,i],seg1[:,:,i])
        eval[i][1] = MyMaritnIndex10(ImageType,ImageSet[:,:,i],seg2[:,:,i])
        eval[i][2] = MyMaritnIndex10(ImageType,ImageSet[:,:,i],seg3[:,:,i])

    return eval

############################################
#FOR TESTING PURPOSES ON OUR 198 IMAGES ONLY
############################################
def batchEvaluateRGB():
 
    #loading cluastering algorithm's output
    kmeans_RGB_321 = np.load('./images/ClusterIm321_MyKmeans_CCIm.npy')
    fcm_RGB_321 = np.load('./images/ClusterIm321_FCM_CCIm.npy')
    som_RGB_321 = np.load('./images/ClusterIm321_MySOM_CCIm.npy')
    gmm_RGB_321 = np.load('./images/ClusterIm321_GMM_CCIm.npy')
    spectral_RGB_321 = np.load('./images/ClusterIm321_MySpectral_CCIm.npy')
    
    kmeans_RGB_481 = np.load('./images/ClusterIm481_MyKmeans_CCIm.npy')
    fcm_RGB_481 = np.load('./images/ClusterIm481_FCM_CCIm.npy')
    som_RGB_481 = np.load('./images/ClusterIm481_MySOM_CCIm.npy')
    gmm_RGB_481 = np.load('./images/ClusterIm481_GMM_CCIm.npy')
    spectral_RGB_481 = np.load('./images/ClusterIm481_MySpectral_CCIm.npy')
    
    seg1_321 = np.load('./images/Seg1_321.npy')
    seg2_321 = np.load('./images/Seg2_321.npy')
    seg3_321 = np.load('./images/Seg3_321.npy')
    
    seg1_481 = np.load('./images/Seg1_481.npy')
    seg2_481 = np.load('./images/Seg2_481.npy')
    seg3_481 = np.load('./images/Seg3_481.npy')
    
    imageType = 'RGB'
    #dispatching for evaluation
    np.save('./evals/rgb/kmeans_321_rgb.npy', dispatchForEval(imageType,kmeans_RGB_321,seg1_321,seg2_321,seg3_321))
    np.save('./evals/rgb/fcm_321_rgb.npy', dispatchForEval(imageType,fcm_RGB_321,seg1_321,seg2_321,seg3_321))
    np.save('./evals/rgb/som_321_rgb.npy', dispatchForEval(imageType,som_RGB_321,seg1_321,seg2_321,seg3_321))
    np.save('./evals/rgb/gmm_321_rgb.npy', dispatchForEval(imageType,gmm_RGB_321,seg1_321,seg2_321,seg3_321))
    np.save('./evals/rgb/spectral_321_rgb.npy', dispatchForEval(imageType,spectral_RGB_321,seg1_321,seg2_321,seg3_321))
    np.save('./evals/rgb/kmeans_481_rgb.npy', dispatchForEval(imageType,kmeans_RGB_481, seg1_481,seg2_481,seg3_481))
    np.save('./evals/rgb/fcm_481_rgb.npy', dispatchForEval(imageType,fcm_RGB_481, seg1_481,seg2_481,seg3_481))
    np.save('./evals/rgb/som_481_rgb.npy', dispatchForEval(imageType,som_RGB_481, seg1_481,seg2_481,seg3_481))
    np.save('./evals/rgb/gmm_481_rgb.npy', dispatchForEval(imageType,gmm_RGB_481, seg1_481,seg2_481,seg3_481))
    np.save('./evals/rgb/spectral_481_rgb.npy', dispatchForEval(imageType,spectral_RGB_481, seg1_481,seg2_481,seg3_481))


############################################
#FOR TESTING PURPOSES ON OUR 198 IMAGES ONLY
############################################
def batchEvaluateHyper():
    
    #loading pavia
    mat_contents = sio.loadmat('./images/hyper/PaviaGrTruth.mat')
    ground_truth = mat_contents['PaviaGrTruth']
    GroundTruthMask = sio.loadmat('./images/hyper/PaviaGrTruthMask.mat')
    GroundTruthMask = GroundTruthMask['PaviaGrTruthMask']
    
    #kmeans
    ClusterIm = np.load('./images/hyper/P_IHYPER_Kmeans.npy')
    ClusterIm = ClusterIm*GroundTruthMask
    res = MyMaritnIndex10('Hyper',ClusterIm,ground_truth)
    print('kmeans')
    print(res)
    
    #fcm
    ClusterIm = np.load('./images/hyper/P_IHYPER_FCM.npy')
    ClusterIm = ClusterIm*GroundTruthMask
    res = MyMaritnIndex10('Hyper',ClusterIm,ground_truth)
    print('fcm')
    print(res)
    
    #som
    ClusterIm = np.load('./images/hyper/P_IHYPER_FCM.npy')
    ClusterIm = ClusterIm*GroundTruthMask
    res = MyMaritnIndex10('Hyper',ClusterIm,ground_truth)
    print('som')
    print(res)
    
    #gmm
    ClusterIm = np.load('./images/hyper/P_IHYPER_GMM.npy')
    ClusterIm = ClusterIm*GroundTruthMask
    res = MyMaritnIndex10('Hyper',ClusterIm,ground_truth)
    print('gmm')
    print(res)
    
    #spectral
    ClusterIm = np.load('./images/hyper/P_IHYPER_Spectral.npy')
    ClusterIm = ClusterIm*GroundTruthMask
    res = MyMaritnIndex10('Hyper',ClusterIm,ground_truth)
    print('gmm')
    print(res)

def main():
    print "Don't call these main, nothing useful, the functions MyMartinIndex10 needs to be called from this script"
    #RGB batch evaluation
    #batchEvaluateRGB()

    #Hyper batch evaluation
    #batchEvaluateHyper()



if __name__ == '__main__':
    main()

