# 嘗試用 patel legnth : patel width 來畫出圖形，看看效果如何。
# 根據 data.DESCR, 如下, 意思好像是說這兩個值有很高的 class correlation, 我猜是族類的代表特徵，做出來看看。
# :Summary Statistics:
#     ============== ==== ==== ======= ===== ====================
#                     Min  Max   Mean    SD   Class Correlation
#     ============== ==== ==== ======= ===== ====================
#     sepal length:   4.3  7.9   5.84   0.83    0.7826
#     sepal width:    2.0  4.4   3.05   0.43   -0.4194
#     petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)
#     petal width:    0.1  2.5   1.20  0.76     0.9565  (high!)
#     ============== ==== ==== ======= ===== ====================

import pdb
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
 
data = load_iris()
y = data.target
X = data.data
# pca = PCA(n_components=2)
# reduced_X = pca.fit_transform(X)
pdb.set_trace() 
 
red_x, red_y = [], []
blue_x, blue_y = [], []
green_x, green_y = [], []
 
for i in range(len(X)):
    if y[i] == 0:
        red_x.append(X[i][3])
        red_y.append(X[i][2])
    elif y[i] == 1:
        blue_x.append(X[i][3])
        blue_y.append(X[i][2])
    else:
        green_x.append(X[i][3])
        green_y.append(X[i][2])
 
plt.scatter(red_x, red_y, c='r', marker='x')
plt.scatter(blue_x, blue_y, c='b', marker='D')
plt.scatter(green_x, green_y, c='g', marker='.')
plt.show()
