from numpy import *


def loadDataSet(fileName):  # 读取含有换行符的数据文件
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float, curLine)  # 将所有元素映射为浮点数
        dataMat.append(fltLine)
    return dataMat


def distEclud(vecA, vecB):  # 计算欧式距离
    return sqrt(sum(power(vecA - vecB, 2)))  # la.norm(vecA-vecB)


def randCent(dataSet, k):  # 为给定数据集构建一个包含k个随机质心的集合
    n = shape(dataSet)[1]  # 特征数
    centroids = mat(zeros((k, n)))  # 质心矩阵
    for j in range(n):  # 随机质心必须要在整个数据集的边界之内
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = mat(minJ + rangeJ * random.rand(k, 1))  # 产生k行1列的随机值
    return centroids


# K-均值聚类算法
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):  # k需要指定；距离默认使用欧氏距离；初始质心默认随机
    m = shape(dataSet)[0]  # 样本数
    clusterAssment = mat(zeros((m, 2)))  # 创建用于分配数据点的矩阵，一列记录索引值，一列存储当前点到簇质心的误差
    centroids = createCent(dataSet, k)
    clusterChanged = True  # 该值为True则继续迭代，为False则停止迭代
    while clusterChanged:
        clusterChanged = False
        for i in range(m):  # 遍历所有数据点
            minDist = inf  # 无穷大infinity
            minIndex = -1  # 最短距离对应的质心
            for j in range(k):  # 遍历k个类，寻找最近的质心
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex: clusterChanged = True  # 改变类别
            clusterAssment[i, :] = minIndex, minDist ** 2
        print(centroids)
        for cent in range(k):  # 重新计算质心
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]  # 获得这个类的所有点
            centroids[cent, :] = mean(ptsInClust, axis=0)  # 按平均值计算新的质心；axis = 0 按列计算
    return centroids, clusterAssment


# 二分K-均值聚类算法：克服K-均值算法收敛于局部最小值的问题
def biKmeans(dataSet, k, distMeas=distEclud):  # 距离默认使用欧氏距离
    m = shape(dataSet)[0]  # 样本数
    clusterAssment = mat(zeros((m, 2)))  # 存储每个点的簇分配结果和平方误差
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList = [centroid0]  # 创建仅含1个质心的列表
    for j in range(m):  # 计算初始误差
        clusterAssment[j, 1] = distMeas(mat(centroid0), dataSet[j, :]) ** 2
    while (len(centList) < k):  # 当簇数目小于指定的k
        lowestSSE = inf  # 无穷大
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]  # 获得当前在簇i中的数据点
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)  # 二分
            sseSplit = sum(splitClustAss[:, 1])  # 计算二分后的这两个簇的SSE
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])  # 计算这两个簇以外的剩余数据集的SSE
            print("sseSplit, and notSplit: ", sseSplit, sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:  # 选取最小的SSE的结果
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        # 当使用KMeans()函数并指定簇为2时，会得到两个编号分别为0和1的结果簇；需要将这些簇编号修改为划分簇及新加簇的编号
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)  # 新加簇
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit  # 划分簇
        print('the bestCentToSplit is: ', bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))
        # 用两个质心代替一个质心
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]
        centList.append(bestNewCents[1, :].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss  # 簇分配结果更新
    return mat(centList), clusterAssment


# Yahoo! PlaceFinder API
import urllib
import json


def geoGrab(stAddress, city):
    apiStem = 'http://where.yahooapis.com/geocode?'  # 为goecoder创建一个字典和常数
    params = {}
    params['flags'] = 'J'  # JSON return type
    params['appid'] = 'aaa0VN6k'
    params['location'] = '%s %s' % (stAddress, city)
    url_params = urllib.urlencode(params)
    yahooApi = apiStem + url_params  # print url_params
    print(yahooApi)
    c = urllib.urlopen(yahooApi)
    return json.loads(c.read())


from time import sleep


def massPlaceFind(fileName):
    fw = open('places.txt', 'w')
    for line in open(fileName).readlines():
        line = line.strip()
        lineArr = line.split('\t')
        retDict = geoGrab(lineArr[1], lineArr[2])
        if retDict['ResultSet']['Error'] == 0:
            lat = float(retDict['ResultSet']['Results'][0]['latitude'])
            lng = float(retDict['ResultSet']['Results'][0]['longitude'])
            print("%s\t%f\t%f" % (lineArr[0], lat, lng))
            fw.write('%s\t%f\t%f\n' % (line, lat, lng))
        else:
            print("error fetching")

        sleep(1)
    fw.close()


# 球面距离计算及簇绘图函数
def distSLC(vecA, vecB):  # Spherical Law of Cosines
    a = sin(vecA[0, 1] * pi / 180) * sin(vecB[0, 1] * pi / 180)
    b = cos(vecA[0, 1] * pi / 180) * cos(vecB[0, 1] * pi / 180) * \
        cos(pi * (vecB[0, 0] - vecA[0, 0]) / 180)
    return arccos(a + b) * 6371.0  # pi is imported with numpy


import matplotlib
import matplotlib.pyplot as plt


def clusterClubs(numClust=5):
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]
    scatterMarkers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:, 0].A == i)[0], :]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:, 0].flatten().A[0], ptsInCurrCluster[:, 1].flatten().A[0], marker=markerStyle,
                    s=90)
    ax1.scatter(myCentroids[:, 0].flatten().A[0], myCentroids[:, 1].flatten().A[0], marker='+', s=300)
    plt.show()
