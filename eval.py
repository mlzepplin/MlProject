#############################################
# Evaluation Code - Need to fix the interface
# and re-check for numeric stability (no overflows)
#############################################

import scipy.io as sio

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
    print "Total Pixels: "+ str(totalPixelInGroundTruth)

    ErrorGS = 0.0
    for j in ClusterPixelCountMapForIg:
        Wj = ClusterPixelCountMapForIg[j] / totalPixelInGroundTruth
        print "Wj for Cluster " + str(j) + " is " + str(Wj)
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


def main():
    mat_contents = sio.loadmat('/home/ani2404/Desktop/mlproject1/PaviaGrTruth.mat')
    ground_truth = mat_contents['PaviaGrTruth']
    mat_contents = sio.loadmat('/home/ani2404/Desktop/mlproject1/hyperout.mat')
    label_image = mat_contents['ClusterIm']
    print MyMaritnIndex('Hyper',label_image,ground_truth)

if __name__ == '__main__':
    main()


