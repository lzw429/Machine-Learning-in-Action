# coding=utf-8
# FP 树的类定义
class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None  # 链接相似的元素项
        self.parent = parentNode
        self.children = {}  # 字典

    def inc(self, numOccur):
        self.count += numOccur

    def disp(self, ind=1):  # 将树以文本形式显示，用于调试
        print '  ' * ind, self.name, ' ', self.count
        for child in self.children.values():
            child.disp(ind + 1)


def createTree(dataSet, minSup=1):  # 从数据集建立 FP-tree；minSup 是最小支持度
    headerTable = {}  # 字典
    # 遍历数据集两次
    for trans in dataSet:  # 第一次遍历计算发生频率；trans 即交易项目
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
            # get 方法返回 item 项的值，如果不存在该项返回0
    for k in headerTable.keys():  # 移除不满足最小支持度的元素项
        if headerTable[k] < minSup:
            del (headerTable[k])
    freqItemSet = set(headerTable.keys())  # 建立集合
    # print 'freqItemSet: ',freqItemSet
    if len(freqItemSet) == 0:  # 如果没有元素项满足最小支持度，即所有项都不频繁，退出
        return None, None
    for k in headerTable:
        headerTable[k] = [headerTable[k], None]  # 使用结点链接重新格式化表头
    # print 'headerTable: ',headerTable
    retTree = treeNode('Null Set', 1, None)  # 建立树
    for tranSet, count in dataSet.items():  # 第二次遍历数据集
        localD = {}
        for item in tranSet:  # 根据全局频率对每个事务中的元素进行排序
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
            updateTree(orderedItems, retTree, headerTable, count)  # 使用排序后的频率项集对树进行填充
    return retTree, headerTable  # 返回树和头指针表


# 树生长
def updateTree(items, inTree, headerTable, count):
    if items[0] in inTree.children:  # 测试事务中的首项元素是否作为子结点存在，如果存在更新该元素项的计数
        inTree.children[items[0]].inc(count)
    else:  # 如果不存在，创建一个新的 treeNode 并将其作为一个子结点添加到树中；
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        if headerTable[items[0]][1] == None:  # 更新头指针表
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:  # call updateTree() with remaining ordered items
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)


# 确保结点链接指向树中该元素项的每一个实例
def updateHeader(nodeToTest, targetNode):  # 该版本不使用递归
    while (nodeToTest.nodeLink != None):  # 不要使用递归遍历链表！
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode


def ascendTree(leafNode, prefixPath):  # 从叶结点上升到根结点
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)


def findPrefixPath(basePat, treeNode):  # treeNode comes from header table
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats


def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1])]  # (sort header table)
    for basePat in bigL:  # 从头指针表的底部开始
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        # print 'finalFrequent Item: ',newFreqSet    #append to set
        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        # print 'condPattBases :',basePat, condPattBases
        # 2. construct cond FP-tree from cond. pattern base
        myCondTree, myHead = createTree(condPattBases, minSup)
        # print 'head from conditional tree: ', myHead
        if myHead != None:  # 3. mine cond. FP-tree
            # print 'conditional tree for: ',newFreqSet
            # myCondTree.disp(1)
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)


# 简单数据集及数据包装器
def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat


def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1  # fronzenset 不可变的集合
    return retDict


import twitter
from time import sleep
import re


def textParse(bigString):
    urlsRemoved = re.sub('(http:[/][/]|www.)([a-z]|[A-Z]|[0-9]|[/.]|[~])*', '', bigString)
    listOfTokens = re.split(r'\W*', urlsRemoved)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def getLotsOfTweets(searchStr):
    CONSUMER_KEY = ''
    CONSUMER_SECRET = ''
    ACCESS_TOKEN_KEY = ''
    ACCESS_TOKEN_SECRET = ''
    api = twitter.Api(consumer_key=CONSUMER_KEY, consumer_secret=CONSUMER_SECRET,
                      access_token_key=ACCESS_TOKEN_KEY,
                      access_token_secret=ACCESS_TOKEN_SECRET)
    # you can get 1500 results 15 pages * 100 per page
    resultsPages = []
    for i in range(1, 15):
        print "fetching page %d" % i
        searchResults = api.GetSearch(searchStr, per_page=100, page=i)
        resultsPages.append(searchResults)
        sleep(6)
    return resultsPages


def mineTweets(tweetArr, minSup=5):
    parsedList = []
    for i in range(14):
        for j in range(100):
            parsedList.append(textParse(tweetArr[i][j].text))
    initSet = createInitSet(parsedList)
    myFPtree, myHeaderTab = createTree(initSet, minSup)
    myFreqList = []
    mineTree(myFPtree, myHeaderTab, minSup, set([]), myFreqList)
    return myFreqList

# minSup = 3
# simpDat = loadSimpDat()
# initSet = createInitSet(simpDat)
# myFPtree, myHeaderTab = createTree(initSet, minSup)
# myFPtree.disp()
# myFreqList = []
# mineTree(myFPtree, myHeaderTab, minSup, set([]), myFreqList)
