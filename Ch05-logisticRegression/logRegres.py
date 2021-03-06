from numpy import *


def loadDataSet():
    dataMat = [];
    labelMat = []  # 分类标签
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()  # strip去除首尾指定字符串，split分割字符串
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


def gradAscent(dataMatIn, classLabels):  # 梯度上升
    dataMatrix = mat(dataMatIn)  # 转化为NumPy矩阵数据类型
    labelMat = mat(classLabels).transpose()  # 矩阵转置，行向量转化为列向量
    m, n = shape(dataMatrix)
    alpha = 0.001  # 学习率
    maxCycles = 500  # 迭代次数
    weights = ones((n, 1))  # 权重列向量
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)  # 列向量
        error = (labelMat - h)
        weights += alpha * dataMatrix.transpose() * error
    return weights


def plotBestFit(weights):  # 画出数据集和Logistic回归最佳拟合直线的函数
    import matlotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)  # 创建二维数组
    n = shape(dataArr)[0]  # dataArr的行数，即样本数量
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i] == 1):
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)  # 行1列1图1
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def stocGradAscent0(dataMatrix, classLabels):  # 随机梯度上升算法
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights += alpha * error * dataMatrix[i]
    return weights


def stocGradAscent1(dataMatrix, classLabels, numIter=150):  # 改进的随机梯度上升算法
    m, n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = range(m)  # 1到m的顺序序列
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.0001  # alpha随着迭代次数不断减小，但不会减小到0
            randIndex = int(random.uniform(0, len(dataIndex)))  # 随机选取更新，随机数范围是0到len
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights += alpha * error * dataMatrix[randIndex]
            del (dataIndex[randIndex])  # 删掉该随机值
    return weights


def classifyVector(inX, weights):  # 感知器算法
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colicTest():  # 打开训练集和测试集
    frTrain = open('horseColicTraining.txt')  # 训练集
    frTest = open('horseColicTest.txt')  # 测试集
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')  # 以制表符分割
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)  # trainingSet是lineArr组成的列表，lineArr是数字组成的列表
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)  # 计算回归系数向量
    errorCount = 0;
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0  # 计算测试集样本数
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)  # 模型分类错误率
    print("the error rate of this test is: %f" % errorRate)
    return errorRate


def multiTest():  # 计算numTests次求分类错误率的平均值
    numTests = 10;  # 计算次数
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum / float(numTests)))
