
--- marker ---

<py>
    import numpy as np
    from sklearn.datasets import load_iris
    from sklearn import tree

    iris = load_iris()
    test_idx = [0,50,100]  # 挑三個 row 正好是每一種 iris 的開頭第一個

    # training data 
    # 從原始 array 裡剔掉 3 個，剩下的用來當 training 組
    train_target = np.delete(iris.target, test_idx)
    
    # data 就是 features; target 就是 label
    train_data = np.delete(iris.data, test_idx, axis=0)

    # testing data 
    # 從原始 array 裡取 3 個用來當 testing 組
    test_target = iris.target[test_idx]
    test_data = iris.data[test_idx]

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_data, train_target)

    print(test_target)
    print(clf.predict(test_data))
    
    # 結果正確！
    # G:\Google developers {Machine Learning}>python test-iris.py
    # [0 1 2]
    # [0 1 2]
    # 
    # G:\Google developers {Machine Learning}>
    
    push(locals())
</py> 
constant locals // ( -- dict ) static values of the locals() of the above namespace

cr locals :> ['iris']['DESCR'] . \ 查看 iris dataset 的說明

stop 

    '''
    >>> train_target = np.delete(iris.target, test_idx)
    >>> iris.target
    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2
           ])
    >>> len(train_target)
    147
    >>> len(iris.target)
    150
    >>> 
    >>> train_target
    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, # 少一個
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
           1, 1, 1, # 少一個
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2  
           ])

    '''
