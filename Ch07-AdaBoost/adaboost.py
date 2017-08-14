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
                errArr[predictedVals == labelMat] = 0  # 如果预测正确，误差是0
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

#
def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m, 1)) / m)  # init D to all equal
    aggClassEst = mat(zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)  # build Stump
        # print "D:",D.T
        alpha = float(
            0.5 * log((1.0 - error) / max(error, 1e-16)))  # calc alpha, throw in max(error,eps) to account for error=0
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)  # store Stump Params in Array
        # print "classEst: ",classEst.T
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)  # exponent for D calc, getting messy
        D = multiply(D, exp(expon))  # Calc New D for next iteration
        D = D / D.sum()
        # calc training error of all classifiers, if this is 0 quit for loop early (use break)
        aggClassEst += alpha * classEst
        # print "aggClassEst: ",aggClassEst.T
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print "total error: ", errorRate
        if errorRate == 0.0: break
    return weakClassArr, aggClassEst


def adaClassify(datToClass, classifierArr):
    dataMatrix = mat(datToClass)  # do stuff similar to last aggClassEst in adaBoostTrainDS
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], \
                                 classifierArr[i]['thresh'], \
                                 classifierArr[i]['ineq'])  # call stump classify
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print aggClassEst
    return sign(aggClassEst)


def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0, 1.0)  # cursor
    ySum = 0.0  # variable to calculate AUC
    numPosClas = sum(array(classLabels) == 1.0)
    yStep = 1 / float(numPosClas);
    xStep = 1 / float(len(classLabels) - numPosClas)
    sortedIndicies = predStrengths.argsort()  # get sorted index, it's reverse
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    # loop through all the values, drawing a line segment at each point
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0;
            delY = yStep;
        else:
            delX = xStep;
            delY = 0;
            ySum += cur[1]
        # draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c='b')
        cur = (cur[0] - delX, cur[1] - delY)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False positive rate');
    plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print "the Area Under the Curve is: ", ySum * xStep
