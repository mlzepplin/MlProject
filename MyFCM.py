# coding=utf-8
from __future__ import division, print_function
import numpy as np
import skfuzzy as fuzz


def MyFCM(Im, ImageType, k):
    # pre_processing
    if ImageType == 'RGB':
        r, c = Im.shape[0:2]
        data = np.zeros((3, r * c))
        #    r,c,p = self._weights.shape[0:3]
        #    weights = np.zeros((r*c,p))
        n = -1
        for i in range(r):
            for j in range(c):
                n = n + 1
                data[:, n] = Im[i, j, :]
    gybicsc = np.zeros(k)
    fpcs = []
    for i1 in range(2, k + 2):
        _, _, _, _, _, _, fpc = fuzz.cluster.cmeans(data, i1, 2, error=0.005, maxiter=1000, init=None)
        fpcs.append(fpc)

    index = np.argwhere(fpcs == max(fpcs))
    if len(index) > 1:
        index = min(index)
    index = index + 2
    index = int(index)
    _, u, _, _, _, _, _ = fuzz.cluster.cmeans(data, index, 2, error=0.005, maxiter=1000, init=None)
    labels = np.argmax(u, axis=0)
    labels = labels + 1
    ClusterIm = np.zeros((r, c))  # 注意是float，是否改成uint8
    n2 = -1
    for i2 in range(r):
        for j2 in range(c):
            n2 = n2 + 1
            ClusterIm[i2, j2] = labels[n2]
    return ClusterIm  # 目前只返回这个 CCIm最后再统一写个函数转换
