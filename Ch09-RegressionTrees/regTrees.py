# coding=utf-8
from numpy import *


# CART算法的实现
def loadDataSet(fileName):
    dataMat = []  # 假定最后一列是目标值
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')  # 以制表符为标志，分离出当前行上的元素
        fltLine = map(float, curLine)  # 所有元素转化为float型
        dataMat.append(fltLine)
    return dataMat


# 二分数据集合
def binSplitDataSet(dataSet, feature, value):  # 数据集合，待切分的特征，该特征的某个值
    # 在给定特征和特征值的情况下，该函数通过数组过滤的方式将上述数据集合切分得到两个子集并返回
    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :][0]  # 最后的[0]是取选中的行的所有数据？
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :][0]
    return mat0, mat1


def createTree(dataSet, leafType=regLeaf, errType=regErr,
               ops=(1, 4)):  # leafType是建立叶节点的函数引用，errType是误差计算函数引用，ops是包含树构建所需其他参数的元组；这是一个递归函数
    # 假定数据集是Numpy矩阵，可对数组进行过滤
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)  # 选择最佳划分
    if feat == None:
        return val  # 如果划分时触发了停止条件，会返回val
    retTree = {}  # 字典
    retTree['spInd'] = feat  # 划分标准
    retTree['spVal'] = val  # 划分值
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)  # 左子树
    retTree['right'] = createTree(rSet, leafType, errType, ops)  # 右子树
    return retTree


# 回归树的切分函数
def regLeaf(dataSet):  # 返回被用于每一片叶的值
    return mean(dataSet[:, -1])  # 目标变量的均值


def regErr(dataSet):
    return var(dataSet[:, -1]) * shape(dataSet)[0]  # 均方差 * 样本数 = 总方差


def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    # 在构建树时考虑生成叶的终止条件就是预剪枝，预剪枝对tolS和tolN非常敏感。
    tolS = ops[0]  # 容许的误差下降值
    tolN = ops[1]  # 切分的最少样本数
    # 退出条件1：如果所有的目标值都是相同的；set()是构建集合
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)  # 直接创建叶节点，叶节点的值返回None
    m, n = shape(dataSet)
    # 最佳特征的选择是通过选取最小的残差平方和(residual sum of squares)
    S = errType(dataSet)  # 当前数据集的误差，将用于后面的退出条件2
    bestS = inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n - 1):  # 遍历所有特征；最后一列是目标值
        for splitVal in set(dataSet[:, featIndex]):  # 遍历某一特征中所有可能出现的值
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  # 两类当中有一类比切分的最少样本数更小，不划分
                continue
            newS = errType(mat0) + errType(mat1)  # 新切分误差
            if newS < bestS:
                bestIndex = featIndex  # 更新划分的特征
                bestValue = splitVal  # 更新划分阈值
                bestS = newS  # 更新划分误差
    # 退出条件2：如果误差的下降(S-beatS)比误差下降阈值tolS更小，则更新划分方式的意义不大
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    # 退出条件3：如果执行到此的结果中，两类当中有一类比切分的最少样本数更小，不划分
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex, bestValue  # 返回最佳划分特征和划分的最佳阈值


# 后剪枝：回归树剪枝函数
def isTree(obj):  # 判断当前处理的节点是否是叶节点，即判断它是不是字典类型的变量
    return (type(obj).__name__ == 'dict')


def getMean(tree):  # 递归函数，遍历tree，若找到两个叶节点则计算它们的平均值。
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0


def prune(tree, testData):  # tree是待剪枝的树，testData是剪枝所需的测试数据
    # 如果测试集是空的，求这个树的平均值
    if shape(testData)[0] == 0:
        return getMean(tree)
    # 如果有一个分支是树，递归剪枝
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        if isTree(tree['left']):
            tree['left'] = prune(tree['left'], lSet)
        if isTree(tree['right']):
            tree['right'] = prune(tree['right'], rSet)
    # 如果分支都是叶，尝试合并它们
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:, -1] - tree['left'], 2)) + \
                       sum(power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = sum(power(testData[:, -1] - treeMean, 2))
        # 比较合并前后的误差以决定是否合并
        if errorMerge < errorNoMerge:
            print "merging"
            return treeMean
        else:
            return tree
    # 如果分支中有树，由于这是递归算法，子树是已被后剪枝过的，直接返回
    else:
        return tree


# 模型树的叶节点生成函数
def linearSolve(dataSet):  # 会被其他两个函数调用的辅助函数；将数据集格式化成目标变量Y和自变量X
    m, n = shape(dataSet)
    X = mat(ones((m, n)))
    Y = mat(ones((m, 1)))
    # 数据格式化，第0列是1，作为偏置
    X[:, 1:n] = dataSet[:, 0:n - 1]
    Y = dataSet[:, -1]  # 分离出Y
    xTx = X.T * X
    if linalg.det(xTx) == 0.0:  # 行列式为0
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')  # 矩阵是奇异阵，不可逆，无法通过正规方程组求权重
    ws = xTx.I * (X.T * Y)  # 正规方程组求权重
    return ws, X, Y


def modelLeaf(dataSet):  # 建立线性模型并返回权重
    ws, X, Y = linearSolve(dataSet)
    return ws


def modelErr(dataSet):  # 在给定数据集上计算误差
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat, 2))


# 用树回归进行预测的代码
def regTreeEval(model, inDat):
    return float(model)


def modelTreeEval(model, inDat):
    n = shape(inDat)[1]  # 特征数量
    X = mat(ones((1, n + 1)))
    X[:, 1:n + 1] = inDat  # 第0列作为偏置
    return float(X * model)


def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree):  # 是叶
        return modelEval(tree, inData)  # 计算预测值
    if inData[tree['spInd']] > tree['spVal']:  # 如果输入数据的划分标准的那个特征的数值大于划分阈值
        if isTree(tree['left']):  # 左分支是子树
            return treeForeCast(tree['left'], inData, modelEval)
        else:  # 左分支是叶节点
            return modelEval(tree['left'], inData)
    else:  # 如果输入数据的划分标准的那个特征的数值小于等于划分阈值
        if isTree(tree['right']):  # 右分支是子树
            return treeForeCast(tree['right'], inData, modelEval)
        else:  # 右分支是叶节点
            return modelEval(tree['right'], inData)


def createForeCast(tree, testData, modelEval=regTreeEval):	# 以向量形式返回一组预测值
    m = len(testData)
    yHat = mat(zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat
