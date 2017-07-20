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
    data = np.array(df)  # df.shape is (747,8) data.shape too.
    feature = data[:,:7]    # all rows, column 0..7, feature.shape --> (747,7)
    label = data[:,7]    # all rows, column 7, label.shape --> (747,)
    return feature, label

def load_data(data_path):
    df = pd.read_csv(data_path)
    data = np.array(df)  # df.shape is (747,8) data.shape too.
    feature = data[:,:7]    # all rows, column 0..7, feature.shape --> (747,7)
    label = data[:,7]    # all rows, column 7, label.shape --> (747,)
    return feature, label
'''

def load_data(data_path):
    df = pd.read_csv(data_path, sep="\s*,", skipinitialspace=True)
    data = np.array(df)  # df.shape is (747,8) data.shape too.
    feature = data[:,:7]    # all rows, column 0..7, feature.shape --> (747,7)
    label = data[:,7]    # all rows, column 7, label.shape --> (747,)
    return feature, label

def evaluation(clf):
    result = []
    for i in range(100):     # 做 10 次，每次都亂取 30% 來對照
        x_train, x_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.3)
        clf.fit(x_train, y_train)
        result.append(np.mean(y_test == clf.predict(x_test)))    # predict 的結果與 y_test 比對，看答對的比例
    print(result)
    print("Minimum %f" % np.array(result).min())
    print("Maximum %f" % np.array(result).max())
    
# if __name__ == '期末大作业1':  # was '__main__': 在 PyCharm 好像就得這樣。
if __name__ == '__main__':
    X, y = load_data("wh300.csv")
    
    # 把 feature 中的 string 都盡量換成 numerical code 幫助 ML
    number = preprocessing.LabelEncoder()
    # number.fit_transform(df.Customer)
    # number.fit_transform(X[:,0]) # Customer
    # number.fit_transform(X[:,1]) # DEPT
    # number.fit_transform(X[:,2]) # ProjectName
    # number.fit_transform(X[:,3]) # PartNo
    # number.fit_transform(X[:,4]) # Date(Min)
    # number.fit_transform(X[:,5]) # QTY
    # number.fit_transform(X[:,6]) # Days(Max)
    # number.fit_transform(X[:,7]) # Tag

    X[:,0] = number.fit_transform(X[:,0]) # Customer
    X[:,1] = number.fit_transform(X[:,1]) # DEPT
    X[:,2] = number.fit_transform(X[:,2]) # ProjectName
    X[:,3] = number.fit_transform(X[:,3]) # PartNo

    X = np.array(X, dtype=np.int64)
    y = np.array(y, dtype=np.int64)

    print("\n---- Decision Tree ----")
    evaluation(DecisionTreeClassifier())
    print("\n---- KNN ----")
    evaluation(KNeighborsClassifier())
    print("\n---- GaussianNB ----")
    evaluation(GaussianNB())
    