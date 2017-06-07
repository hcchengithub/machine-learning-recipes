from sklearn.datasets import load_iris
iris = load_iris()
print(iris.feature_names)
print(iris.target_names)
iris.data[0]
iris.target[0]
for i in range(len(iris.target)):
	print ("Example %d: label %s, features %s" % (i, iris.target[i], iris.data[i]))
