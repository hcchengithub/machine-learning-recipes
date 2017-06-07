import numpy as np
import sklearn.cluster as skc
from sklearn import metrics
import matplotlib.pyplot as plt
import pdb

mac2id=dict()
onlinetimes=[]
f=open('TestData.txt',encoding='utf-8')
for line in f:
    mac=line.split(',')[2]
    onlinetime=int(line.split(',')[6])
    starttime=int(line.split(',')[4].split(' ')[1].split(':')[0])
    if mac not in mac2id:
        mac2id[mac]=len(onlinetimes)
        onlinetimes.append((starttime,onlinetime))
    else:
        onlinetimes[mac2id[mac]]=[(starttime,onlinetime)]
        pdb.set_trace() # mac address 沒有重複的，本段虛設。有的話後面的蓋前面，簡化的作法。

real_X=np.array(onlinetimes).reshape((-1,2))

X=np.log(1+real_X[:,1:]) # 改成 log 好像太簡單了，把大數字區壓扁正好符合本場合
db=skc.DBSCAN(eps=0.14,min_samples=10).fit(X)
labels = db.labels_
pdb.set_trace() 
 
print('Labels:')
print(labels)
ratio=len(labels[labels[:] == -1]) / len(labels)
print('Noise ratio:',format(ratio, '.2%'))
 
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
 
print('Estimated number of clusters: %d' % n_clusters_)
print("Silhouette Coefficient: %0.3f"% metrics.silhouette_score(X, labels))
 
for i in range(n_clusters_):
    print('Cluster ',i,':')
    count = len(X[labels == i])
    mean = np.mean(real_X[labels ==i][:,1])
    std = np.std(real_X[labels ==i][:,1])
    print('\t number of sample: ',count)
    print('\t mean of sample : ',format(mean,'.1f'))
    print('\t std of sample : ',format(std,'.1f'))
     
# pdb.set_trace()

plt.hist(X,24)
plt.show()



