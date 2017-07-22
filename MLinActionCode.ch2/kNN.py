from numpy import *
import operator
from os import listdir


# k近邻算法
def classify0(inX, dataSet, labels, k):  # 监督学习，labels是给定的
    dataSetSize = dataSet.shape[0]  # dataSet行数，即样本数

    # 计算欧几里得距离
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # tile函数将inX重复dataSetSize行，1列
    sqDiffMat = diffMat ** 2  # 平方
    sqDistances = sqDiffMat.sum(axis=1)  # 不写axis是全部相加，axis=0按列相加，axis=1按行相加
    distances = sqDistances ** 0.5  # 开方
    sortedDistIndicies = distances.argsort()  # argsort排序

    # 选择距离最小的k个点
    classCount = {}  # 字典
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]  # 得到第i近的点的类别
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # 此类的点的个数+1；如果值不在字典中，get函数返回0
    # 字典迭代器；按字典第二个元素的次序；逆序排序
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]  # [0][0]发生频率最高的元素标签，[0][1]发生频率最高的元素在字典中的值


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# 将文本记录转换为Numpy的解析程序
def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())  # 得到文件行数
    returnMat = zeros((numberOfLines, 3))  # 将返回的矩阵
    classLabelVector = []  # 将返回的标签向量
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


# 归一化Normalization特征值
def autoNorm(dataSet):
    minVals = dataSet.min(0)  # 0表示从列中选取最小值
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals  # ranges是行向量
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]  # 数据集行数
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))  # 特征值相除
    return normDataSet, ranges, minVals
