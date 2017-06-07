import pdb
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import cross_validation
 
data=pd.read_csv('000777.csv',encoding='gbk',parse_dates=[0],index_col=0)
# pdb.set_trace() # data 本來是新的在上面
data.sort_index(0,ascending=True,inplace=True) # 排序 sort 過後變成最舊的在上面

# raw data has 20 years long, but here only analysis the last 150 days of a chosen day
dayfeature=150
featurenum=5*dayfeature  # 取五個有關係的欄位，特徵值。
x=np.zeros((data.shape[0]-dayfeature,featurenum+1))
# (Pdb) data.shape[0] 等於是 len(data) --> # 4752
# (Pdb) (data.shape[0]-dayfeature,featurenum+1)
# (4602, 751)
# (Pdb) np.zeros((data.shape[0]-dayfeature,featurenum+1))  弄出個都是 0 的 matrix
# array([[ 0.,  0.,  0., ...,  0.,  0.,  0.],
#        [ 0.,  0.,  0., ...,  0.,  0.,  0.],
#        [ 0.,  0.,  0., ...,  0.,  0.,  0.],
#        ...,
#        [ 0.,  0.,  0., ...,  0.,  0.,  0.],
#        [ 0.,  0.,  0., ...,  0.,  0.,  0.],
#        [ 0.,  0.,  0., ...,  0.,  0.,  0.]])
# (Pdb) np.sum(np.zeros((data.shape[0]-dayfeature,featurenum+1))) --> 0.0 果然都是 0
y=np.zeros((data.shape[0]-dayfeature))
 
# pdb.set_trace() # 斷點設在下面兩行,觀察 x array 的值是哪來的
for i in range(0,data.shape[0]-dayfeature):
    x[i,0:featurenum]=np.array(data[i:i+dayfeature] \
          [[u'收盘价',u'最高价',u'最低价',u'开盘价',u'成交量']]).reshape((1,featurenum))
    x[i,featurenum]=data.ix[i+dayfeature][u'开盘价']
 
for i in range(0,data.shape[0]-dayfeature):
    if data.ix[i+dayfeature][u'收盘价']>=data.ix[i+dayfeature][u'开盘价']:
        y[i]=1
    else:
        y[i]=0          
# pdb.set_trace() # feature x, lable y 

clf=svm.SVC(kernel='poly')  # so what is SVM? See "11-提交-监督学习 - 课程导学.pdf"
result = []
for i in range(10):
    x_train, x_test, y_train, y_test = \
                cross_validation.train_test_split(x, y, test_size = 0.2)
    clf.fit(x_train, y_train)
    result.append(np.mean(y_test == clf.predict(x_test)))
print("svm classifier accuacy:")
print(result)



