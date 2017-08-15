# coding=utf-8
from numpy import *
from numpy import linalg as la  # 线性代数


def loadExData():
    return [[0, 0, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1],
            [1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 1, 0, 0]]


def loadExData2():
    return [[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
            [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
            [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
            [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
            [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
            [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
            [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
            [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
            [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
            [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]


# 相似度计算
def ecludSim(inA, inB):  # 定义：相似度= 1 / (1 + 欧氏距离)
    return 1.0 / (1.0 + la.norm(inA - inB))


def pearsSim(inA, inB):  # 皮尔逊相关系数，取值范围归一化到0到1之间
    if len(inA) < 3: return 1.0  # 检查是否存在3个或更多的点，若不存在返回1.0，因为两个向量完全相关
    return 0.5 + 0.5 * corrcoef(inA, inB)[0][1]


def cosSim(inA, inB):  # 余弦相似度，取值范围归一化到0到1之间
    num = float(inA.T * inB)
    denom = la.norm(inA) * la.norm(inB)
    return 0.5 + 0.5 * (num / denom)  # 余弦相似度 = 向量内积 / 两向量2范数之积


# 基于物品相似度的推荐引擎
# 计算在给定相似度计算方法的条件下，用户对物品的估计评分值
def standEst(dataMat, user, simMeas, item):  # 数据矩阵，用户编号，相似度计算方法，物品编号
    # dataSet 行对应用户，列对应物品
    n = shape(dataMat)[1]  # n是物品数目
    simTotal = 0.0
    ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user, j]  # 用户评分
        if userRating == 0:
            continue  # 没有评分则下一轮循环
        # 寻找两个都被评级的物品，使用了逻辑与
        overLap = nonzero(logical_and(dataMat[:, item].A > 0, dataMat[:, j].A > 0))[0]
        if len(overLap) == 0:  # 两者没有任何重合元素
            similarity = 0  # 相似度为0
        else:
            similarity = simMeas(dataMat[overLap, item], dataMat[overLap, j])
        print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal


# 推荐引擎
def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):  # 产生最高的N个推荐结果
    unratedItems = nonzero(dataMat[user, :].A == 0)[1]  # 寻找未评级物品
    if len(unratedItems) == 0:
        return 'you rated everything'
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]


# 基于SVD的评分估计
def svdEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    U, Sigma, VT = la.svd(dataMat)
    Sig4 = mat(eye(4) * Sigma[:4])  # arrange Sig4 into a diagonal matrix
    xformedItems = dataMat.T * U[:, :4] * Sig4.I  # create transformed items
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 or j == item: continue
        similarity = simMeas(xformedItems[item, :].T, xformedItems[j, :].T)
        print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal


# 图像压缩函数
def printMat(inMat, thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i, k]) > thresh:
                print
                1,
            else:
                print
            0,
        print
        ''


def imgCompress(numSV=3, thresh=0.8):
    myl = []
    for line in open('0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = mat(myl)
    print("****original matrix******")
    printMat(myMat, thresh)
    U, Sigma, VT = la.svd(myMat)
    SigRecon = mat(zeros((numSV, numSV)))
    for k in range(numSV):  # construct diagonal matrix from vector
        SigRecon[k, k] = Sigma[k]
    reconMat = U[:, :numSV] * SigRecon * VT[:numSV, :]
    print("****reconstructed matrix using %d singular values******" % numSV)
    printMat(reconMat, thresh)
