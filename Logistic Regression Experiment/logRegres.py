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
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights


def plotBestFit(weights):  # 画出数据集和Logistic回归最佳拟合直线的函数
    import matlotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
