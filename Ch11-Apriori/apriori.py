from numpy import *


def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


# 构建集合C1，C1是大小为1的所有候选项集的集合
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])

    C1.sort()
    return map(frozenset, C1)  # 使用frozen，可以作为字典键值使用


def scanD(D, Ck, minSupport):  # 数据集D，候选项集列表Ck，最小支持度minSupport
    ssCnt = {}  # 字典
    for tid in D:  # 遍历数据集中的交易，transaction in dataset
        for can in Ck:  # 遍历候选项
            if can.issubset(tid):  # 如果can 是tid 的子集
                if not ssCnt.has_key(can):  # 如果ssCnt没有can这个键
                    ssCnt[can] = 1  # 字典中新建一个键
                else:
                    ssCnt[can] += 1
    numItems = float(len(D))  # 数据集长度
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key] / numItems  # 计算支持度
        if support >= minSupport:
            retList.insert(0, key)  # 在列表首部插入新集合
        supportData[key] = support
    return retList, supportData


# Apriori算法
def aprioriGen(Lk, k):  # 输入参数是频繁项集列表Lk与项集元素个数k，输出为合并得到的候选项集Ck
    retList = []  # 列表
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            L1 = list(Lk[i])[:k - 2]  # 取前k-2项
            L2 = list(Lk[j])[:k - 2]
            L1.sort()
            L2.sort()
            if L1 == L2:  # 如果两个集合的前面k-2个元素都相等
                retList.append(Lk[i] | Lk[j])  # 并集
    return retList


def apriori(dataSet, minSupport=0.5):  # 默认最小支持度是0.5
    C1 = createC1(dataSet)
    D = map(set, dataSet)  # 将set()映射到dataSet列表中的每一项
    L1, supportData = scanD(D, C1, minSupport)  # 从候选项集C1找频繁项集L1
    L = [L1]  # 将L1放入列表L中，L将包含L1、L2、L3
    k = 2
    while (len(L[k - 2]) > 0):
        Ck = aprioriGen(L[k - 2], k)  # 由频繁项集L[k-2]生成候选项集Ck
        Lk, supK = scanD(D, Ck, minSupport)  # 扫描数据集Ck生成Lk
        supportData.update(supK)  # 丢掉不满足最小支持度要求的项集
        L.append(Lk)  # Lk添加到列表L
        k += 1
    return L, supportData


# 关联规则生成函数
def generateRules(L, supportData, minConf=0.7):
    # L是频繁项集列表；supportData 是由scanD得来的存储支持度的字典；minConf是最小可信度阈值
    bigRuleList = []  # 基于可信度（confidence）排序，规则存放在bigRuleList中
    for i in range(1, len(L)):  # 只获取有两个或更多元素的集合
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):  # 如果频繁项集数目超过2
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:  # 如果项集中只有两个元素
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList


# 计算可信度
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = []  # 创建一个用于返回的列表
    for conseq in H:
        conf = supportData[freqSet] / supportData[freqSet - conseq]  # 计算可信度
        if conf >= minConf:
            print(freqSet - conseq, '-->', conseq, 'conf:', conf)
            brl.append((freqSet - conseq, conseq, conf))  # brl 是前面通过检查的bigRuleList
            prunedH.append(conseq)
    return prunedH


def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    if (len(freqSet) > (m + 1)):  # 尝试进一步合并
        Hmp1 = aprioriGen(H, m + 1)
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if (len(Hmp1) > 1):  # 至少有两个集合用于合并
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)


from time import sleep
from votesmart import votesmart

votesmart.apikey = 'a7fa40adec6f4a77178799fae4441030'


# votesmart.apikey = 'get your api key first'
def getActionIds():
    actionIdList = [];
    billTitleList = []
    fr = open('recent20bills.txt')
    for line in fr.readlines():
        billNum = int(line.split('\t')[0])
        try:
            billDetail = votesmart.votes.getBill(billNum)  # api call
            for action in billDetail.actions:
                if action.level == 'House' and \
                        (action.stage == 'Passage' or action.stage == 'Amendment Vote'):
                    actionId = int(action.actionId)
                    print('bill: %d has actionId: %d' % (billNum, actionId))
                    actionIdList.append(actionId)
                    billTitleList.append(line.strip().split('\t')[1])
        except:
            print("problem getting bill %d" % billNum)
        sleep(1)  # delay to be polite
    return actionIdList, billTitleList


def getTransList(actionIdList, billTitleList):  # this will return a list of lists containing ints
    itemMeaning = ['Republican', 'Democratic']  # list of what each item stands for
    for billTitle in billTitleList:  # fill up itemMeaning list
        itemMeaning.append('%s -- Nay' % billTitle)
        itemMeaning.append('%s -- Yea' % billTitle)
    transDict = {}  # list of items in each transaction (politician)
    voteCount = 2
    for actionId in actionIdList:
        sleep(3)
        print('getting votes for actionId: %d' % actionId)
        try:
            voteList = votesmart.votes.getBillActionVotes(actionId)
            for vote in voteList:
                if not transDict.has_key(vote.candidateName):
                    transDict[vote.candidateName] = []
                    if vote.officeParties == 'Democratic':
                        transDict[vote.candidateName].append(1)
                    elif vote.officeParties == 'Republican':
                        transDict[vote.candidateName].append(0)
                if vote.action == 'Nay':
                    transDict[vote.candidateName].append(voteCount)
                elif vote.action == 'Yea':
                    transDict[vote.candidateName].append(voteCount + 1)
        except:
            print("problem getting actionId: %d" % actionId)
        voteCount += 2
    return transDict, itemMeaning
