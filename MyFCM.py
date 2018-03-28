import numpy as np


def initWithFuzzyMat(r, c, k):
    fuzzyMat = np.zeros((k, r, c))
    for rowIndex in range(r):
        for colIndex in range(c):
            memDegreeSum = 0
            randoms = np.random.rand(k - 1, 1)
            for kIndex in range(k - 1):
                fuzzyMat[kIndex, rowIndex, colIndex] = randoms[kIndex, 0] * (1 - memDegreeSum)
                memDegreeSum += fuzzyMat[kIndex, rowIndex, colIndex]
            fuzzyMat[-1, rowIndex, colIndex] = 1 - memDegreeSum
    # print(fuzzyMat)
    return fuzzyMat


# there are k centers with m different values
def calCentWithFuzzyMat(dataSet, fuzzyMat, p):
    r, c, m = dataSet.shape
    k = fuzzyMat.shape[0]
    centroids = np.mat(np.zeros((k, m)))
    for kIndex in range(k):
        degExpArray = np.power(fuzzyMat[kIndex, :], p)
        denominator = np.sum(degExpArray)
        numerator = np.array(np.zeros((1, m)))
        for rowIndex in range(r):
            for colIndex in range(c):
                numerator += dataSet[rowIndex, colIndex] * degExpArray[rowIndex, colIndex]
        centroids[kIndex, :] = numerator / denominator
    return centroids


def eculidDistance(vectA, vectB):
    return np.sqrt(np.sum(np.power(vectA - vectB, 2)))


def calFuzzyMatWithCent(dataSet, centroids, p):
    r, c, m = dataSet.shape
    ce = centroids.shape[0]
    fuzzyMat = np.zeros((ce, r, c))
    for ceIndex in range(ce):
        for rowIndex in range(r):
            for colIndex in range(c):
                d_ij = eculidDistance(centroids[ceIndex, :], dataSet[rowIndex, colIndex, :])
                fuzzyMat[ceIndex, rowIndex, colIndex] = 1 / np.sum(
                    [np.power(d_ij / eculidDistance(centroid, dataSet[rowIndex, colIndex, :]), 2 / (p - 1)) for centroid
                     in
                     centroids])
    return fuzzyMat


def calTargetFunc(dataSet, fuzzyMat, centroids, k, p):
    r, c, m = dataSet.shape
    ce = fuzzyMat.shape[0]
    targetFunc = 0
    for ceIndex in range(ce):
        for rowIndex in range(r):
            for colIndex in range(c):
                targetFunc += eculidDistance(centroids[ceIndex, :], dataSet[rowIndex, colIndex, :]) ** 2 * np.power(
                    fuzzyMat[ceIndex, rowIndex, colIndex], p)
    return targetFunc


# we assume that the input Im is a r*c*3 matrix, in which every node in r*c matrix represent a rgb node
def FCM(Im, NumClusts):
    r, c, m = Im.shape
    fuzzyMat = initWithFuzzyMat(r, c, NumClusts)
    centroids = calCentWithFuzzyMat(Im, fuzzyMat, 2)
    targetFunc = 0
    while targetFunc * 0.99 > calTargetFunc(Im, fuzzyMat, centroids, NumClusts, 2):
        fuzzyMat = calFuzzyMatWithCent(Im, centroids, 2)
        centroids = calCentWithFuzzyMat(Im, fuzzyMat, 2)
    return fuzzyMat, centroids


def MyFCM(Im, ImType, NumClusts):
    fuzzyMat, centroids = FCM(Im, NumClusts)
    r, c = Im.shape[0:2]
    ClusterIm = np.zeros((r, c))
    for rowIndex in range(r):
        for colIndex in range(c):
            temp = np.array(fuzzyMat[:, rowIndex, colIndex]).tolist()
            index = temp.index(max(temp))+1
            ClusterIm[rowIndex, colIndex] = index
    return ClusterIm
