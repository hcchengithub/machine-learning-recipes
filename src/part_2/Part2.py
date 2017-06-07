# Part 2 - Visualizing a Decision Tree - https://youtu.be/tNa99PG8hR8

# Build one on a real dataset, add code to visualize it, and practice reading it - so 
# you can see how it works under the hood.

# Use Iris flower data set: https://en.wikipedia.org/wiki/Iris_flower_data_set
# Identify type of flower based on measurements
# Dataset includes 3 species of Iris flowers: setosa, versicolor, virginica
# 4 features used to describe: length and width of sepal and petal
# 50 examples of each type for 150 total examples

# Goals
# 1-Import dataset
# 2-Train a classifier
# 3-Predict label for new flower
# 4-Visualize the tree

# scikit-learn datasets: http://scikit-learn.org/stable/datasets/
# already includes Iris dataset: load_iris

from sklearn.datasets import load_iris

iris = load_iris()

# print iris.feature_names  # metadata: names of the features
# print iris.target_names  # metadata: names of the different types of flowers

#Python 3.6.0 |Anaconda 4.3.1
print(iris.feature_names)
print(iris.target_names)

# print iris.data  # features and examples themselves
# print iris.data[0]  # first flower
# print iris.target[0]  # contains the labels

#Python 3.6.0 |Anaconda 4.3.1
print(iris.data[0])
print(iris.target[0])

# print entire dataset
for i in range(len(iris.target)):
    print ("Example %d: label %s, features %s" % (i, iris.target[i], iris.data[i]))

# Testing Data
# Examples used to test the classifier's accuracy
# Not part of the training data

# ＠＠＠＠＠＠＠＠＠＠ 第二部份 ＠＠＠＠＠＠＠＠＠＠
import numpy as np
import pdb
# from sklearn.datasets import load_iris 
from sklearn import tree

iris = load_iris()
# here, we remove the first example of each flower
# found at indices: 0, 50, 100
test_idx = [0, 50, 100]  # 故意挑出三個出來，當作測試組。每種類各挑一個，共三個。其餘的留在訓練組。

# create 2 new sets of variables, for training and testing
# training data
# remove the entires from the data and target variables
train_target = np.delete(iris.target, test_idx)       # 從訓練組中剔除測試組的三筆資料
train_data = np.delete(iris.data, test_idx, axis=0)   # 我的 test-iris.py 有針對 np.delete() 的實驗，搞懂它。

# testing data
test_target = iris.target[test_idx]  # 故意挑出三個出來，當作測試組。每種類各挑一個，共三個。其餘的留在訓練組。
test_data = iris.data[test_idx]

# create new classifier
clf = tree.DecisionTreeClassifier() # 祭出法寶
# train on training data
clf.fit(train_data, train_target) # 施咒，請寶貝轉身！

# what we expect
# print test_target

#Python 3.6.0 |Anaconda 4.3.1
print(test_target)

# what tree predicts
#print clf.predict(test_data)

#Python 3.6.0 |Anaconda 4.3.1
print (clf.predict(test_data))

pdb.set_trace() # breakpoint

# Visualize
# from scikit decision tree tutorial: http://scikit-learn.org/stable/modules/tree.html
from sklearn.externals.six import StringIO
import pydotplus as pydot

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data,
                     feature_names=iris.feature_names,
                     class_names=iris.target_names,
                     filled=True, rounded=True,
                     impurity=False)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
pdb.set_trace()
pass
graph.write_pdf("iris.pdf")
# Image(graph.create_png())

'''
--
graph.write_pdf("iris.pdf") 有問題，研究幾下發現不行了，別人已經問了：
https://stackoverflow.com/questions/38176472/graph-write-pdfiris-pdf-attributeerror-list-object-has-no-attribute-writ

答案可能是：
    option-a  要改用 pydotplus instead of pydot
        仍有後續問題：
        Q. pydotplus.graphviz.InvocationException: GraphViz's executables not found
		A. https://stackoverflow.com/questions/18438997/why-is-pydot-unable-to-find-graphvizs-executables-in-windows-8
		   https://stackoverflow.com/questions/28312534/graphvizs-executables-are-not-found-python-3-4
		   "c:\\Users\hcche\Downloads\windows _ Graphviz - Graph Visualization Software.pdf" 
		   c:\\Users\hcche\Downloads\graphviz-2.38.msi  
			--> Add PATH manually --> path=%PATH%c:\Program Files (x86)\Graphviz2.38\bin
		成功了！ ==> 要改用 pydotplus, 要 install Graphviz, 要為 Graphviz 手動設置 PATH, that's all.

    option-b 改用 graph[0] instead of graph  <--- failed
        > c:\\users\hcche\documents\github\machine-learning-recipes\src\part_2\part2.py(102)<module>()
        -> graph[0].write_pdf("iris.pdf")
        (Pdb) c
        Traceback (most recent call last):
          File "C:\\users\hcche\AppData\Local\Programs\Python\Python36\lib\site-packages\pydot.py", line 1878, in create
            stderr=subprocess.PIPE, stdout=subprocess.PIPE)
          File "C:\\users\hcche\AppData\Local\Programs\Python\Python36\lib\subprocess.py", line 707, in __init__
            restore_signals, start_new_session)
          File "C:\\users\hcche\AppData\Local\Programs\Python\Python36\lib\subprocess.py", line 990, in _execute_child
            startupinfo)
        FileNotFoundError: [WinError 2] The system cannot find the file specified
        During handling of the above exception, another exception occurred:
        Traceback (most recent call last):
          File "Part2.py", line 102, in <module>
            graph[0].write_pdf("iris.pdf")
          File "C:\\users\hcche\AppData\Local\Programs\Python\Python36\lib\site-packages\pydot.py", line 1691, in <lambda>
            self.write(path, format=f, prog=prog))
          File "C:\\users\hcche\AppData\Local\Programs\Python\Python36\lib\site-packages\pydot.py", line 1774, in write
            s = self.create(prog, format)
          File "C:\\users\hcche\AppData\Local\Programs\Python\Python36\lib\site-packages\pydot.py", line 1883, in create
            prog=prog))
        Exception: "dot.exe" not found in path.
--
(Pdb) dir(graph)
['__add__', '__class__', '__contains__', '__delattr__', '__delitem__', '__dir__', 
'__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', 
'__gt__', '__hash__', '__iadd__', '__imul__', '__init__', '__init_subclass__', '__iter__', 
'__le__', '__len__', '__lt__', '__mul__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', 
'__repr__', '__reversed__', '__rmul__', '__setattr__', '__setitem__', '__sizeof__', 
'__str__', '__subclasshook__', 'append', 'clear', 'copy', 'count', 'extend', 'index', 
'insert', 'pop', 'remove', 'reverse', 'sort']
(Pdb)

(Pdb) dot_data.getvalue()
'digraph Tree {\nnode [shape=box, style="filled, rounded", color="black", fontname=helvetica] ;\n
edge [fontname=helvetica] ;\n0 [label="petal length (cm) <= 2.45\\nsamples = 147\\n
value = [49, 49, 49]\\nclass = setosa", fillcolor="#e5813900"] ;\n1 [label="samples = 49\\n
value = [49, 0, 0]\\nclass = setosa", fillcolor="#e58139ff"] ;\n
0 -> 1 [labeldistance=2.5, labelangle=45, headlabel="True"] ;\n
2 [label="petal width (cm) <= 1.75\\nsamples = 98\\nvalue = [0, 49, 49]\\n
class = versicolor", fillcolor="#39e58100"] ;\n
0 -> 2 [labeldistance=2.5, labelangle=-45, headlabel="False"] ;\n
3 [label="petal length (cm) <= 4.95\\nsamples = 53\\nvalue = [0, 48, 5]\\n
class = versicolor", fillcolor="#39e581e4"] ;\n2 -> 3 ;\n
4 [label="petal width (cm) <= 1.65\\nsamples = 47\\nvalue = [0, 46, 1]\\n
class = versicolor", fillcolor="#39e581f9"] ;\n3 -> 4 ;\n5 [label="samples = 46\\n
value = [0, 46, 0]\\nclass = versicolor", fillcolor="#39e581ff"] ;\n4 -> 5 ;\n
6 [label="samples = 1\\nvalue = [0, 0, 1]\\nclass = virginica", fillcolor="#8139e5ff"] ;\n
4 -> 6 ;\n7 [label="petal width (cm) <= 1.55\\nsamples = 6\\nvalue = [0, 2, 4]\\n
class = virginica", fillcolor="#8139e57f"] ;\n3 -> 7 ;\n8 [label="samples = 3\\n
value = [0, 0, 3]\\nclass = virginica", fillcolor="#8139e5ff"] ;\n7 -> 8 ;\n
9 [label="petal length (cm) <= 5.45\\nsamples = 3\\nvalue = [0, 2, 1]\\n
class = versicolor", fillcolor="#39e5817f"] ;\n7 -> 9 ;\n10 [label="samples = 2\\n
value = [0, 2, 0]\\nclass = versicolor", fillcolor="#39e581ff"] ;\n9 -> 10 ;\n
11 [label="samples = 1\\nvalue = [0, 0, 1]\\nclass = virginica", fillcolor="#8139e5ff"] ;\n
9 -> 11 ;\n12 [label="petal length (cm) <= 4.85\\nsamples = 45\\nvalue = [0, 1, 44]\\n
class = virginica", fillcolor="#8139e5f9"] ;\n2 -> 12 ;\n13 [label="sepal length (cm) <= 5.95\\n
samples = 3\\nvalue = [0, 1, 2]\\nclass = virginica", fillcolor="#8139e57f"] ;\n
12 -> 13 ;\n14 [label="samples = 1\\nvalue = [0, 1, 0]\\n
class = versicolor", fillcolor="#39e581ff"] ;\n13 -> 14 ;\n15 [label="samples = 2\\n
value = [0, 0, 2]\\nclass = virginica", fillcolor="#8139e5ff"] ;\n13 -> 15 ;\n
16 [label="samples = 42\\nvalue = [0, 0, 42]\\nclass = virginica", fillcolor="#8139e5ff"] ;\n
12 -> 16 ;\n}'

(Pdb)

'''
