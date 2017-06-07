import pdb
from sklearn import tree
features = [[140,1],[130,1],[150,0],[170,0]]
labels = [0,0,1,1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
pdb.set_trace()

# 故意丟些變化球看看，好玩！
print (clf.predict([[153,0]])) # 1
print (clf.predict([[120,0]])) # 0
print (clf.predict([[190,1]])) # 1
pass
