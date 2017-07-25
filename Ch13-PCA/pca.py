from numpy import *


def loadDataSet(fileName, delim='\t'):  # 这里的loadDataSet使用了两个list comprehension来构建矩阵
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float, line) for line in stringArr]
    return mat(datArr)


def pca(dataMat, topNfeat=9999999):  # 第一个参数是数据集；第二个是可选参数，即应用的N个特征
    # 如果不指定topNfeat值，函数返回前9999999个特征，或原始数据中全部的特征
    meanVals = mean(dataMat, axis=0)  # 按列求平均值
    meanRemoved = dataMat - meanVals  # 减去平均值
    covMat = cov(meanRemoved)  # 计算协方差矩阵
    eigVals, eigVects = linalg.eig(mat(covMat))  # 计算特征值与特征向量
    eigValInd = argsort(eigVals)  # 排序，从小到大
    eigValInd = eigValInd[:-(topNfeat + 1):-1]  # 去除不想保留的维度
    # 上句话使用了切片，从最后一个元素从右向左取到倒数第topNfeat个元素，不包含倒数第(topNfeat+1)个元素
    redEigVects = eigVects[:, eigValInd]  # 得到topNfeat个最大的特征向量
    lowDDataMat = meanRemoved * redEigVects  # 将数据转换到新空间
    reconMat = (lowDDataMat * redEigVects.T) + meanVals  # 重构原始数据用于调试
    return lowDDataMat, reconMat


# 将NaN替换成平均值的函数
def replaceNanWithMean():
    datMat = loadDataSet('secom.data', ' ')
    numFeat = shape(datMat)[1]  # 样本数 = 列数
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:, i].A))[0], i])  # 计算非NaN的平均值
        datMat[nonzero(isnan(datMat[:, i].A))[0], i] = meanVal  # 将所有NaN替换为该平均值
    return datMat
