# coding=utf-8
from numpy import *


# 装载单层决策树数据样本
def loadSimpData():
    datMat = matrix([[1., 2.1],
                     [2., 1.1],
                     [1.3, 1.],
                     [1., 1.],
                     [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels


def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))  # 获取特征数量
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


# 单层决策树生成函数
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):  # 数据集，维度，阈值，与阈值的不等关系
    # 通过阈值比较对数据分类
    retArray = ones((shape(dataMatrix)[0], 1))  # m × 1
    if threshIneq == 'lt':  # "lt" is short for "less than"
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:  # if threshIneq == 'gt'
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


# 遍历 stumpClassify 函数所有可能的输入值，找到数据集上最佳的单层决策树，也就是弱学习器
def buildStump(dataArr, classLabels, D):
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T  # 转置为列向量
    m, n = shape(dataMatrix)
    numSteps = 10.0  # 用于在所有可能值上进行遍历
    bestStump = {}  # 字典，存储给定权重向量D时得到的最佳单层决策树的相关信息
    bestClasEst = mat(zeros((m, 1)))  # 最佳类别估计
    minError = inf  # 最小错误率，初始化为无穷大
    for i in range(n):  # 遍历所有特征
        rangeMin = dataMatrix[:, i].min()  # 该特征中的最小值
        rangeMax = dataMatrix[:, i].max()  # 该特征中的最大值
        stepSize = (rangeMax - rangeMin) / numSteps  # 计算步长
        for j in range(-1, int(numSteps) + 1):  # 在当前特征中遍历所有值
            for inequal in ['lt', 'gt']:  # 从 less than 到 greater than
                threshVal = (rangeMin + float(j) * stepSize)  # 将阈值的范围设置到了整个取值范围之外
                predictedVals = stumpClassify(dataMatrix, i, threshVal,
                                              inequal)  # 调用单层决策树（决策树桩）
                errArr = mat(ones((m, 1)))  # m行1列
                errArr[predictedVals == labelMat] = 0  # 如果预测正确，错误率是0
                weightedError = D.T * errArr  # 计算加权错误率，其中D是权重向量
                # 输出所有值
                # print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i  # 维度
                    bestStump['thresh'] = threshVal  # 阈值
                    bestStump['ineq'] = inequal  # 数据与阈值的不等关系
    return bestStump, minError, bestClasEst


# 基于单层决策树的AdaBoost训练过程
def adaBoostTrainDS(dataArr, classLabels, numIt=40):  # numIt是迭代次数；DS 是 decision stump
    weakClassArr = []  # 弱分类器
    m = shape(dataArr)[0]  # 样本数量
    D = mat(ones((m, 1)) / m)  # 初始化权重矩阵D，所有值相等
    aggClassEst = mat(zeros((m, 1)))  # 记录每个数据点的类别估计累计值
    for i in range(numIt):  # 运行numIt次或训练错误率为0为止
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)  # 建立决策树桩
        # print "D:",D.T # 可查看迭代过程中的权重值
        alpha = float(
            0.5 * log((1.0 - error) / max(error, 1e-16)))  # alpha是在总分类器中，本次单层决策树输出结果的权重；使用max是防止除0溢出
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)  # 将存有弱分类器参数的 bestStump字典 放入弱分类器数组
        # print "classEst: ",classEst.T
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)  # 计算D需要用的指数
        D = multiply(D, exp(expon))  # 为下一次迭代计算新的权重向量D
        D = D / D.sum()
        # 错误率累加计算
        aggClassEst += alpha * classEst
        # print "aggClassEst: ",aggClassEst.T
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print "total error: ", errorRate
        if errorRate == 0.0: break
    return weakClassArr, aggClassEst


# AdaBoost分类函数：利用已训练的多个弱分类器进行分类
def adaClassify(datToClass, classifierArr):  # datToClass是待分类样例；classifierArr是多个弱分类器组成的数组
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]  # 待分类样例个数
    aggClassEst = mat(zeros((m, 1)))  # 记录每个数据点的类别估计累计值
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], \
                                 classifierArr[i]['thresh'], \
                                 classifierArr[i]['ineq'])  # 调用决策树桩分类器
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print aggClassEst
    return sign(aggClassEst)  # 符号函数


# ROC曲线的绘制及AUC计算函数
def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0, 1.0)  # 光标
    ySum = 0.0  # 用于计算AUC的变量
    numPosClas = sum(array(classLabels) == 1.0)  # 这是正例个数；负例个数可由此计算
    yStep = 1 / float(numPosClas)
    xStep = 1 / float(len(classLabels) - numPosClas)
    sortedIndicies = predStrengths.argsort()  # 得到排好序的索引，是逆序的
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    # 遍历所有值，在每个点绘制一条线段
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]
        # 从 cur 划线到 (cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c='b')
        cur = (cur[0] - delX, cur[1] - delY)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print "the Area Under the Curve is: ", ySum * xStep
