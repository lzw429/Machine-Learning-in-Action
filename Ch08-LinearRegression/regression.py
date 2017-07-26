from numpy import *
from time import sleep
import json
import urllib2


# 数据导入函数
def loadDataSet(fileName):  # 打开一个含有分隔符的文本文件
    numFeat = len(open(fileName).readline().split('\t')) - 1  # 获得特征数，减1是因为最后一列是因变量
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))  # 将每个数字读入lineArr
        dataMat.append(lineArr)  # 将每个样本读入dataMat
        labelMat.append(float(curLine[-1]))  # curLine最后一个元素读入labelMat
    return dataMat, labelMat


# 标准回归函数：正规方程（Normal Equation）计算最佳拟合直线
def standRegres(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T  # 转置为列向量
    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0.0:  # 计算xTx的行列式
        print("This matrix is singular, cannot do inverse")  # 这是奇异阵，不可逆
        return  # xTx是奇异阵，无法计算
    ws = xTx.I * (xMat.T * yMat)  # .I是求逆；计算得到回归系数
    return ws


# 局部加权线性回归函数：此处使用高斯核，k是高斯核中的参数；与testPoint越近，权重会越大
# 与kNN一样，该加权模型认为样本点距离越近，越可能符合同一个线性模型
# 注意区分此处的权重weights和回归系数ws，回归系数的计算中加入了权重
def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = mat(xArr)
    yMat = mat(yArr).T  # 转置为列向量
    m = shape(xMat)[0]  # 样本个数
    weights = mat(eye((m)))  # m阶对角权重矩阵
    for j in range(m):  # 下面两行创建权重矩阵
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = exp(diffMat * diffMat.T / (-2.0 * k ** 2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:  # 如果xTx的行列式为0
        print("This matrix is singular, cannot do inverse")  # 这是奇异阵，不可逆
        return  # xTx是奇异阵，无法计算
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws


def lwlrTest(testArr, xArr, yArr, k=1.0):  # 遍历数据点，尝试对每个点都适用lwlr，这有助于求解k的大小
    m = shape(testArr)[0]  # 样本数
    yHat = zeros(m)  # 预测值
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat


def lwlrTestPlot(xArr, yArr, k=1.0):  # 与lwlrTest唯一的不同是先对X排序
    yHat = zeros(shape(yArr))  # 对画图更容易
    xCopy = mat(xArr)
    xCopy.sort(0)
    for i in range(shape(xArr)[0]):
        yHat[i] = lwlr(xCopy[i], xArr, yArr, k)
    return yHat, xCopy


def rssError(yArr, yHatArr):  # 需要yArr和yHatArr都是数组
    return ((yArr - yHatArr) ** 2).sum()  # 最小二乘法计算代价函数


# 岭回归
def ridgeRegres(xMat, yMat, lam=0.2):  # lam是单位矩阵前的系数；lambda是Python关键字，此处使用lam代替
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1]) * lam
    if linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")  # 如果lam是0，denom仍是奇异阵，无法计算
        return
    ws = denom.I * (xMat.T * yMat)
    return ws


def ridgeTest(xArr, yArr):  # 用一组lambda测试结果
    xMat = mat(xArr)
    yMat = mat(yArr).T  # 转置为列向量
    yMean = mean(yMat, 0)  # 每列求平均值
    yMat = yMat - yMean
    # 对特征做标准化处理
    xMeans = mean(xMat, 0)  # 每列求平均值
    xVar = var(xMat, 0)  # 每列求方差
    xMat = (xMat - xMeans) / xVar  # 标准化计算
    numTestPts = 30  # 在30个不同的lambda下调用ridgeRegres
    wMat = zeros((numTestPts, shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, exp(i - 10))
        wMat[i, :] = ws.T
    return wMat


def regularize(xMat):  # 标准化处理
    inMat = xMat.copy()  # 必须使用copy，否则得到索引
    inMeans = mean(inMat, 0)  # 计算平均值
    inVar = var(inMat, 0)  # 计算方差
    inMat = (inMat - inMeans) / inVar  # 标准化
    return inMat


# 前向逐步线性回归：与lasso做法相近但计算简单
def stageWise(xArr, yArr, eps=0.01, numIt=100):  # eps是每次迭代需要调整的步长；numIt表示迭代次数
    xMat = mat(xArr)
    yMat = mat(yArr).T  # 转置为列向量
    yMean = mean(yMat, 0)
    yMat = yMat - yMean  # 也可以使ys标准化，但会减小相关系数
    xMat = regularize(xMat)
    m, n = shape(xMat)
    returnMat = zeros((numIt, n))  # 每次迭代都打印w向量，用于分析算法执行的过程和效果
    ws = zeros((n, 1))
    wsTest = ws.copy()  # 必须使用.copy()，否则得到的是ws的索引
    wsMax = ws.copy()
    for i in range(numIt):  # 贪心算法，每一步尽可能减小误差
        lowestError = inf  # 无穷大infinity
        for j in range(n):  # 对于每个特征
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A)  # 计算平方误差
                if rssE < lowestError:  # 比较，取最小误差
                    lowestError = rssE
                    wsMax = wsTest  # 最小误差时的ws
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat


# 购物信息的获取函数
def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
    sleep(10)
    myAPIstr = 'AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'
    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (
        myAPIstr, setNum)
    pg = urllib2.urlopen(searchURL)
    retDict = json.loads(pg.read())
    for i in range(len(retDict['items'])):
        try:
            currItem = retDict['items'][i]
            if currItem['product']['condition'] == 'new':
                newFlag = 1
            else:
                newFlag = 0
            listOfInv = currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice = item['price']
                if sellingPrice > origPrc * 0.5:
                    print("%d\t%d\t%d\t%f\t%f" % (yr, numPce, newFlag, origPrc, sellingPrice))

                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        except:
            print('problem with item %d' % i)


def setDataCollect():
    scrapePage('setHtml/lego8288.html', 'out.txt', 2006, 800, 49.99)
    scrapePage('setHtml/lego10030.html', 'out.txt', 2002, 3096, 269.99)
    scrapePage('setHtml/lego10179.html', 'out.txt', 2007, 5195, 499.99)
    scrapePage('setHtml/lego10181.html', 'out.txt', 2007, 3428, 199.99)
    scrapePage('setHtml/lego10189.html', 'out.txt', 2008, 5922, 299.99)
    scrapePage('setHtml/lego10196.html', 'out.txt', 2009, 3263, 249.99)


# 交叉验证测试岭回归
def crossValidation(xArr, yArr, numVal=10):  # numVal是交叉验证的次数
    m = len(yArr)  # 样本个数
    indexList = range(m)  # [1,2,...,m]
    errorMat = zeros((numVal, 30))  # 误差矩阵，numVal行30列
    for i in range(numVal):
        trainX = []  # 训练集容器
        trainY = []
        testX = []  # 测试集容器
        testY = []
        random.shuffle(indexList)  # 对indexList进行混洗
        for j in range(m):  # 以indexList前90%的值建立训练集
            if j < m * 0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:  # 剩下10%作为测试集
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = ridgeTest(trainX, trainY)  # 从ridgeRegression得到30个回归系数
        for k in range(30):  # ridgeTest()使用30个不同的lambda创建了30组不同的回归系数
            matTestX = mat(testX)
            matTrainX = mat(trainX)
            meanTrain = mean(matTrainX, 0)
            varTrain = var(matTrainX, 0)
            matTestX = (matTestX - meanTrain) / varTrain  # 训练集标准化
            yEst = matTestX * mat(wMat[k, :]).T + mean(trainY)
            errorMat[i, k] = rssError(yEst.T.A, array(testY))
    meanErrors = mean(errorMat, 0)  # 按列计算30组回归系数的平均误差
    minMean = float(min(meanErrors))
    bestWeights = wMat[nonzero(meanErrors == minMean)]  # nonzero获得索引，找到最优回归系数
    # 建立模型可不标准化
    # 当标准化 Xreg = (x-meanX)/var(x)
    # 或不标准化:  x*w/var(x) - meanX/var(x) +meanY
    xMat = mat(xArr)
    yMat = mat(yArr).T
    meanX = mean(xMat, 0)
    varX = var(xMat, 0)
    # 岭回归使用了数据标准化，而standRegres()没有，为了将上述比较可视化还需将数据还原
    unReg = bestWeights / varX
    print("the best model from Ridge Regression is:\n", unReg)
    print("with constant term: ", -1 * sum(multiply(meanX, unReg)) + mean(yMat))
