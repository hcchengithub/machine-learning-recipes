import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn import cross_validation
import pdb


'''		
def strip(text):
    try:
        return text.strip()
    except AttributeError:
        return text
def load_data(data_path):
	df = pd.read_csv(data_path,
		converters = {
		'Customer' : strip,
		'DEPT' : strip,
		'Project-Name' : strip,
		'PartNo' : strip,
		})
	data = np.array(df)	 # df.shape is (747,8) data.shape too.
	feature = data[:,:7]	# all rows, column 0..7, feature.shape --> (747,7)
	label = data[:,7]	 # all rows, column 7, label.shape --> (747,)
	return feature, label

def load_data(data_path):
	df = pd.read_csv(data_path)
	data = np.array(df)	 # df.shape is (747,8) data.shape too.
	feature = data[:,:7]	# all rows, column 0..7, feature.shape --> (747,7)
	label = data[:,7]	 # all rows, column 7, label.shape --> (747,)
	return feature, label
'''

def load_data(data_path):
	df = pd.read_csv(data_path, sep="\s*,", skipinitialspace=True)
	data = np.array(df)	 # df.shape is (747,8) data.shape too.
	pdb.set_trace() # 888333
	feature = data[:,:7]	# all rows, column 0..7, feature.shape --> (747,7)
	label = data[:,7]	 # all rows, column 7, label.shape --> (747,)
	return feature, label

# if __name__ == '期末大作业1':	# was '__main__': 在 PyCharm 好像就得這樣。
if __name__ == '__main__':
	x_train, y_train = load_data("wh300.csv")
	x_test = x_train # 等於是察看 recall rate
	
	# 把 feature 中的 string 都盡量換成 number 幫助 ML
	number = preprocessing.LabelEncoder()
	# number.fit_transform(df.Customer)
	# number.fit_transform(x_train[:,0]) # Customer
	# number.fit_transform(x_train[:,1]) # DEPT
	# number.fit_transform(x_train[:,2]) # ProjectName
	# number.fit_transform(x_train[:,3]) # PartNo
	# number.fit_transform(x_train[:,4]) # Date(Min)
	# number.fit_transform(x_train[:,5]) # QTY
	# number.fit_transform(x_train[:,6]) # Days(Max)
	# number.fit_transform(x_train[:,7]) # Tag

	x_train[:,3] = number.fit_transform(x_train[:,3]) # PartNo
	x_train[:,2] = number.fit_transform(x_train[:,2]) # ProjectName
	x_train[:,1] = number.fit_transform(x_train[:,1]) # DEPT
	x_train[:,0] = number.fit_transform(x_train[:,0]) # Customer

	x_train = np.array(x_train, dtype=np.int64)
	y_train = np.array(y_train, dtype=np.int64)
 
	print('Start training knn')
	pdb.set_trace()
	knn = KNeighborsClassifier().fit(x_train, y_train)
	print('Training done')
	answer_knn = knn.predict(x_test)  # answer_knn --> array([2, 3, 1, ..., 1, 2, 3], dtype=int64), answer_knn.shape --> (174304,)
	print('Prediction done')

	print('Start training DT')
	dt = DecisionTreeClassifier().fit(x_train, y_train)
	print('Training done')
	answer_dt = dt.predict(x_test)
	print('Prediction done')

	print('Start training Bayes')
	gnb = GaussianNB().fit(x_train, y_train)
	print('Training done')
	answer_gnb = gnb.predict(x_test)
	print('Prediction done')
	np.savetxt("model_1.txt",answer_knn,fmt='%i')
	np.savetxt("model_2.txt",answer_dt,fmt='%i')
	np.savetxt("model_3.txt",answer_gnb,fmt='%i')
	print('model_1.txt model_2.txt model_3.txt have saved. For knn, dt, gnb respectively')

	evaluate = []  # evaluation table
	for i in range(len(answer_knn)):
		evaluate.append([1, 1, 1])	# init
		# knn
		if answer_knn[i]==answer_dt[i]:
			evaluate[i][0] += 1
		if answer_knn[i]==answer_gnb[i]:
			evaluate[i][0] += 1
		# dt
		if answer_dt[i]==answer_knn[i]:
			evaluate[i][1] += 1
		if answer_dt[i]==answer_gnb[i]:
			evaluate[i][1] += 1
		# gnb
		if answer_gnb[i]==answer_knn[i]:
			evaluate[i][2] += 1
		if answer_gnb[i]==answer_dt[i]:
			evaluate[i][2] += 1
	evalu = np.array(evaluate)	# don't know how to sum() without np
	knn_score = np.sum(evalu[:, 0])
	dt_score = np.sum(evalu[:, 1])
	gnb_score = np.sum(evalu[:, 2])
	print('Comparison, the higher score the higher consistancy:')
	print('knn: %i, dt: %i, gnb: %i' % (knn_score, dt_score, gnb_score))

	evaluate = np.array([["knn", "dt", "gnb"]]+evaluate)
	np.savetxt("eval.csv", evaluate, fmt="%s", delimiter=',')
	print('eval.csv has saved.')

	