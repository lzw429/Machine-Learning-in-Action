from math import log
import operator


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    # change to discrete values
    return dataSet, labels


# 计算给定数据集的信息熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}  # 创建新的字典
    for featVec in dataSet:  # 为所有可能的分类创建字典
        currentLabel = featVec[-1]  # 记录标签
        if currentLabel not in labelCounts.keys():  # 如果标签不在字典的键中
            labelCounts[currentLabel] = 0  # 扩展字典将当前键值加入字典
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0  # 信息熵
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries  # 通过频率计算概率
        shannonEnt -= prob * log(prob, 2)  # 底数为2
    return shannonEnt


# 按照给定特征划分数据集
def splitDataSet(dataSet, axis, value):  # 待划分的数据集、划分数据集的特征、需要返回的特征的值
    retDataSet = []  # 创建新的列表
    for featVec in dataSet:
        if featVec[axis] == value:  # 如果第axis个特征是value
            reducedFeatVec = featVec[:axis]  # 抽取用于划分数据集的特征，[:axis]不包含第axis个
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


# 选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # 最后一列是标签，numFeatures是特征个数
    baseEntropy = calcShannonEnt(dataSet)  # 经验熵
    bestInfoGain = 0.0  # 初始的信息增益
    bestFeature = -1
    for i in range(numFeatures):  # 遍历数据集中所有特征
        featList = [example[i] for example in dataSet]  # 某个特征的所有值的列表
        uniqueVals = set(featList)  # 集合具有互异性
        newEntropy = 0.0  # 该特征的经验条件熵
        for value in uniqueVals:  # 计算一个特征内各种值的经验条件熵
            subDataSet = splitDataSet(dataSet, i, value)  # 第i个特征是value 的数据构成的列表
            prob = len(subDataSet) / float(len(dataSet))  # 计算概率值
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy  # 计算信息增益，即经验熵与经验条件熵之差
        if (infoGain > bestInfoGain):  # 计算最好的信息增益比
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature  # 返回一个整数


# 出现次数最多的分类
def majorityCnt(classList):
    classCount = {}  # 字典
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0  # 创建键值为classList中唯一值的数据字典，字典对象存储了classList中每个类标签出现的频率
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)  # 排序字典
    return sortedClassCount[0][0]  # 返回出现次数最多的分类名称


# 创建树的函数代码
def createTree(dataSet, labels):  # 算法本身不需要labels，为了给出数据明确的含义在此作为输入参数
    classList = [example[-1] for example in dataSet]  # 标签构成的列表
    if classList.count(classList[0]) == len(classList):  # 类别完全相同则停止继续划分
        return classList[0]
    if len(dataSet[0]) == 1:  # 遍历完所有特征时
        return majorityCnt(classList)  # 返回出现次数最多的类别
    bestFeat = chooseBestFeatureToSplit(dataSet)  # 选择最好的数据划分方式
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}  # 字典myTree将存储树的所有信息
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]  # 去除bestFeat后的标签，作为子树的标签
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)  # 递归构建决策树
    return myTree


# 使用决策树的分类函数
def classify(inputTree, featLabels, testVec):  # testVec是用于测试的一组数据
    firstStr = inputTree.keys()[0]  # 键
    secondDict = inputTree[firstStr]  # 值
    featIndex = featLabels.index(firstStr)  # 找到特征的位置
    key = testVec[featIndex]  # testVec变量中的值
    valueOfFeat = secondDict[key]  # 子树或树的叶
    if isinstance(valueOfFeat, dict):  # 判断valueOfFeat是否为字典
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:  # 若valueOfFeat不是字典，找到树的叶
        classLabel = valueOfFeat
    return classLabel


# 使用pickle模块存储决策树到硬盘
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
