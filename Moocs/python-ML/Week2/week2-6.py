import numpy as np     #?入numpy工具包
from os import listdir #使用listdir模?，用于??本地文件
from sklearn import neighbors
 
def img2vector(fileName):    
    retMat = np.zeros([1024],int) #定?返回的矩?，大小?1*1024
    fr = open(fileName)           #打?包含32*32大小的?字文件 
    lines = fr.readlines()        #?取文件的所有行
    for i in range(32):           #遍?文件所有行
        for j in range(32):       #并?01?字存放在retMat中     
            retMat[i*32+j] = lines[i][j]    
    return retMat
 
def readDataSet(path):    
    fileList = listdir(path)    #?取文件?下的所有文件 
    numFiles = len(fileList)    #??需要?取的文件的?目
    dataSet = np.zeros([numFiles,1024],int)    #用于存放所有的?字文件
    hwLabels = np.zeros([numFiles])#用于存放??的??(与神?网?的不同)
    for i in range(numFiles):      #遍?所有的文件
        filePath = fileList[i]     #?取文件名?/路?   
        digit = int(filePath.split('_')[0])   #通?文件名?取??     
        hwLabels[i] = digit        #直接存放?字，并非one-hot向量
        dataSet[i] = img2vector(path +'/'+filePath)    #?取文件?容 
    return dataSet,hwLabels
 
#read dataSet
train_dataSet, train_hwLabels = readDataSet('trainingDigits')
knn = neighbors.KNeighborsClassifier(algorithm='kd_tree', n_neighbors=3)
knn.fit(train_dataSet, train_hwLabels)
 
#read  testing dataSet
dataSet,hwLabels = readDataSet('testDigits')
 
res = knn.predict(dataSet)  #???集?行??
error_num = np.sum(res != hwLabels) #??分???的?目
num = len(dataSet)          #??集的?目
print("Total num:",num," Wrong num:", \
      error_num,"  WrongRate:",error_num / float(num))