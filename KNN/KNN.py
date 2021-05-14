import numpy as np
import operator
from os import listdir

def classify0(inX, dataSet, labels, k):
    size = len(dataSet)
    diffMat = np.tile(inX,(size,1))-dataSet
    sqDiffMat = diffMat**2
    sqDistance = np.sum(sqDiffMat,axis=1)
    distances = sqDistance**0.5
    sortedDistIndicies = distances.argsort()
    classCount={}
    i = 0
    while i<k:
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
        i+=1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels =['A','A','B','B']
    return group,labels

group ,lables = createDataSet()
a = classify0([1.1,0.9],group,lables,2)

filename = 'datingTestSet.txt'
def loadData(filename):
    fr = open(filename)
    arrayOnlines = fr.readlines()
    datasize = len(arrayOnlines)
    returnMat = np.zeros((datasize,3))
    classLabels = []
    i = 0
    for line in arrayOnlines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[i,:]=listFromLine[0:3]
        if listFromLine[-1]=='didntLike':
            classLabels.append(1)
        elif listFromLine[-1]=='smallDoses':
            classLabels.append(2)
        elif listFromLine[-1]=='largeDoses':
            classLabels.append(3)
        i+=1
    return returnMat,classLabels

dataMat,datalabels = loadData(filename)

def autoNorm(dataMat):
    max = np.max(dataMat,axis=0)
    min = np.min(dataMat,axis=0)
    range = max-min
    m = len(dataMat)
    normDataSet = dataMat-np.tile(min,(m,1))
    normDataSet = normDataSet/np.tile(range,(m,1))
    return normDataSet,range,min

normMat,range,minval = autoNorm(dataMat)
# print(normMat[0:100,:])
def datingclassTest():
    ratio = 0.1
    dataMat,datalabels = loadData(filename)
    normMat,range,minval = autoNorm(dataMat)
    dataSize = len(normMat)
    testNum = int(dataSize*ratio)
    errorCount = 0.0
    i = 0
    while i <testNum:
        res = classify0(normMat[i, :], normMat[testNum:dataSize, :], datalabels[testNum:dataSize], 3)
        if res != datalabels[i]:
            errorCount += 1
        i+=1
    print('error rate:%f' % (errorCount/float(testNum)))

# datingclassTest()

def img2Vector(filename):
    returnMat = np.zeros((1,1024))
    fr = open(filename)
    i=0
    while i<32:
        lineStr = fr.readline()
        j = 0
        while j <32:
            returnMat[0,32*i+j]=int(lineStr[i])
            j+=1
        i+=1
    return returnMat

def loadDigits():
    trainList = listdir('dataSet/trainingDigits')
    traindataSize = len(trainList)
    hwLabels = []
    trainMat = np.zeros((traindataSize,1024))
    i = 0
    while i<traindataSize:
        fileNameStr = trainList[i]
        fileStr = int(fileNameStr.split('_')[0])
        hwLabels.append(fileStr)
        trainMat[i, :] = img2Vector('dataSet/trainingDigits/%s' % fileNameStr)
        i+=1
    testList = listdir('dataSet/testDigits')
    testdataSize = len(testList)
    testMat = np.zeros((testdataSize,1024))
    errorCount = 0.0
    i=0
    while i <testdataSize:
        fileNameStr = testList[i]
        fileStr = int(fileNameStr.split('_')[0])
        res = classify0(img2Vector('dataSet/testDigits/%s' % fileNameStr),trainMat,hwLabels,3)
        if res != fileStr:
            errorCount+=1.0
        i+=1
    print('error rate:%f' %(errorCount/float(testdataSize)))

loadDigits()
