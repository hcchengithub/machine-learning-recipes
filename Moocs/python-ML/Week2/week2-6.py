import numpy as np     #?�Jnumpy�u��]
from os import listdir #�ϥ�listdir��?�A�Τ_??���a���
from sklearn import neighbors
 
def img2vector(fileName):    
    retMat = np.zeros([1024],int) #�w?��^���x?�A�j�p?1*1024
    fr = open(fileName)           #��?�]�t32*32�j�p��?�r��� 
    lines = fr.readlines()        #?����󪺩Ҧ���
    for i in range(32):           #�M?���Ҧ���
        for j in range(32):       #�}?01?�r�s��bretMat��     
            retMat[i*32+j] = lines[i][j]    
    return retMat
 
def readDataSet(path):    
    fileList = listdir(path)    #?�����?�U���Ҧ���� 
    numFiles = len(fileList)    #??�ݭn?�������?��
    dataSet = np.zeros([numFiles,1024],int)    #�Τ_�s��Ҧ���?�r���
    hwLabels = np.zeros([numFiles])#�Τ_�s��??��??(�O��?�I?�����P)
    for i in range(numFiles):      #�M?�Ҧ������
        filePath = fileList[i]     #?�����W?/��?   
        digit = int(filePath.split('_')[0])   #�q?���W?��??     
        hwLabels[i] = digit        #�����s��?�r�A�}�Done-hot�V�q
        dataSet[i] = img2vector(path +'/'+filePath)    #?�����?�e 
    return dataSet,hwLabels
 
#read dataSet
train_dataSet, train_hwLabels = readDataSet('trainingDigits')
knn = neighbors.KNeighborsClassifier(algorithm='kd_tree', n_neighbors=3)
knn.fit(train_dataSet, train_hwLabels)
 
#read  testing dataSet
dataSet,hwLabels = readDataSet('testDigits')
 
res = knn.predict(dataSet)  #???��?��??
error_num = np.sum(res != hwLabels) #??��???��?��
num = len(dataSet)          #??����?��
print("Total num:",num," Wrong num:", \
      error_num,"  WrongRate:",error_num / float(num))