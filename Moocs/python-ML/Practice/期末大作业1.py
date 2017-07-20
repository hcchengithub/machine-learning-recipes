import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import pdb

def load_train(data_path):
    df = pd.read_table(data_path, delimiter=' ', header=None)  # pandas.core.frame.DataFrame
    data = np.array(df)  # (406708, 55)
    feature = data[:, :54]  # all rows, column 0..53, feature.shape --> (406708, 54)
    label = data[:, 54]  # all rows, column 54, label.shape --> (406708,)
    return feature, label


def load_test(data_path):
    df = pd.read_table(data_path, delimiter=' ', header=None)  # pandas data_frame
    feature = np.array(df)  #
    return feature


# if __name__ == '期末大作业1':  # was '__main__': 在 PyCharm 好像就得這樣。
if __name__ == '__main__': 
    ''' 数据路径 '''  # data/data_train.txt data/data_test.txt
    ''' 读入数据  '''
    x_train, y_train = load_train("data/data_train.txt")  # x_train.shape --> (406708, 54), y_train.shape --> (406708,)
    x_test = load_test("data/data_test.txt")  #  x_test.shape --> (174304, 54)
    print('Start training knn')
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

    '''
        data_test.txt 沒有 label 因此不知道對錯。嘗試把以上三組 predicted 結果比對，看看結果。
        依三個 classifier 意見都相同、都不同、兩個相同，等情形作個分類。如下：

        knn   dt    gnb
        ----  ---   ---
        3     3     3   三個 classifier 意見相同
        2     2     1   兩個 classifier 意見相同，一個不同
        1     2     2   兩個 classifier 意見相同，一個不同
        2     1     2   兩個 classifier 意見相同，一個不同
        1     1     1   三個 classifier 意見都不相同 
		
		各自的總分越高似乎代表成績越好。
    '''
    evaluate = []  # evaluation table
    for i in range(len(answer_knn)):
        evaluate.append([1, 1, 1])  # init
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
    evalu = np.array(evaluate)  # don't know how to sum() without np
    knn_score = np.sum(evalu[:, 0])
    dt_score = np.sum(evalu[:, 1])
    gnb_score = np.sum(evalu[:, 2])
    print('Comparison, the higher score the higher consistancy:')
    print('knn: %i, dt: %i, gnb: %i' % (knn_score, dt_score, gnb_score))

    evaluate = np.array([["knn", "dt", "gnb"]]+evaluate)
    np.savetxt("eval.csv", evaluate, fmt="%s", delimiter=',')
    print('eval.csv has saved.')

	