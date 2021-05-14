import numpy as np

def createDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],  # 切分的词条
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1代表侮辱性文字，0代表正常言论
    return postingList, classVec

def createVocabList(postingList):
    returnSet = set([])
    for line in postingList:
        returnSet = returnSet| set(line)
    return list(returnSet)

def setOfWord2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

def classify0(voabMat, classVec):
    numOfAb = sum(classVec)
    classSize = len(classVec)
    numOfWord = len(voabMat[0])
    pAb = float(numOfAb)/float(classSize)#不当言论的概率
    p0Vec = np.ones(numOfWord)
    p1Vec = np.ones(numOfWord)
    p0Num = 2.0
    p1Num = 2.0
    for i in classVec:
        if i==0:
            p0Vec+=voabMat[i]
            p0Num+=sum(voabMat[i])
        else:
            p1Vec+=voabMat[i]
            p1Num+=sum(voabMat[i])
    p1Vect = np.log(p1Vec/p1Num)
    p0Vect = np.log(p0Vec/p0Num)
    return pAb,p1Vect,p0Vect

def classifyNB(inputVec,p0Vect,p1Vect,pAb):
    p1 = sum(p1Vect*inputVec)+np.log(pAb)
    p0 = sum(p0Vect*inputVec)+np.log(1-pAb)
    if p1>p0:
        return 1
    else:
        return 0

if __name__=='__main__':
    postingList, classVec = createDataSet()
    myVocabList = createVocabList(postingList)
    print(len(myVocabList))
    vocabMat = []
    for line in postingList:
        vocabMat.append(setOfWord2Vec(myVocabList,line))
    pAb,p1Vect,p0Vect = classify0(vocabMat,classVec)
    testEntry = ['stupid','garbage']
    thisDoc = setOfWord2Vec(myVocabList,testEntry)
    print(classifyNB(thisDoc,p0Vect,p1Vect,pAb))