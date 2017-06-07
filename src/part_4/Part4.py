# Let's Write a Pipeline - Machine Learning Recipes #4 - https://youtu.be/84gqSbLcBFE

# How to test a model and determine accuracy

# Partition data into 2 sets, train and test

# import a dataset
from sklearn import datasets

iris = datasets.load_iris()

# Can think of classifier as a function f(x) = y
X = iris.data  # features
y = iris.target  # labels

# partition into training and testing sets
from sklearn.cross_validation import train_test_split

# test_size=0.5 -> split in half
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# Classifier
from sklearn import tree

my_classifier = tree.DecisionTreeClassifier()
my_classifier.fit(X_train, y_train)

# predict
predictions = my_classifier.predict(X_test)
print (predictions)

# test
from sklearn.metrics import accuracy_score

print (accuracy_score(y_test, predictions))

'''
Results:
c:\Users\hcche\Documents\GitHub\machine-learning-recipes\src\part_4>python Part4.py
C:\Users\hcche\AppData\Local\Programs\Python\Python36\lib\site-packages\sklearn\cross_validation.py:44: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
  "This module will be removed in 0.20.", DeprecationWarning)
[2 0 2 0 1 0 2 2 0 1 0 2 1 0 1 1 2 2 0 1 1 0 1 2 1 0 0 2 0 0 2 1 0 2 2 2 1  <------ predictions
 1 2 0 2 1 1 0 0 1 2 1 2 1 2 2 2 1 0 0 0 2 1 1 1 0 1 0 0 2 0 2 0 1 1 0 1 1
 2]
0.946666666667 <--------- accuracy_score

> c:\users\hcche\documents\github\machine-learning-recipes\src\part_4\part4.py(42)<module>()
-> from sklearn.neighbors import KNeighborsClassifier
(Pdb)

'''

import pdb
pdb.set_trace() # Breakpoint

# Repeat using KNN
# Classifier
from sklearn.neighbors import KNeighborsClassifier

my_classifier = KNeighborsClassifier()
my_classifier.fit(X_train, y_train)

# predict
predictions = my_classifier.predict(X_test)
print (predictions)

# test
from sklearn.metrics import accuracy_score

print (accuracy_score(y_test, predictions))

> c:\users\hcche\documents\github\machine-learning-recipes\src\part_4\part4.py(42)<module>()
-> from sklearn.neighbors import KNeighborsClassifier
(Pdb) c

'''
Results:
[2 0 2 0 1 0 2 2 0 1 0 2 1 0 2 1 2 2 0 1 1 0 1 2 1 0 0 2 0 0 2 1 0 2 2 2 1   <------ predictions of KNeighbors
 1 2 0 2 1 1 0 0 1 2 1 2 1 2 2 2 2 0 0 0 2 2 1 1 0 1 0 0 2 0 2 0 1 1 0 1 1
 2]
0.96 <--------- A better accuracy_score !!
'''

