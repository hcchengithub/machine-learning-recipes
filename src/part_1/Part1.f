
\ How to run :
\ ^D 之後整個 copy-paste 給 peforth interpreter 或 include path/part1.f 即可

--- marker --- 
<py>
    # Part 1 - Hello World - https://youtu.be/cKxRvEZd3Mw

    # Follow a recipe for supervised learning (a technique to create a classifier from examples) and code it up.

    from sklearn import tree

    # Examples
    # Weight Texture Label
    # 150g   Bumpy   Orange
    # 170g   Bumpy   Orange
    # 140g   Smooth  Apple
    # 130g   Smooth  Apple

    # Training Data
    # features = [[140, "smooth"], [130, "smooth"], [150, "bumpy"], [170, "bumpy"]]  # Input to classifier
    features = [[140, 1], [130, 1], [150, 0], [170, 0]]  # scikit-learn uses real-valued features
    # labels = ["apple", "apple", "orange", "orange"]  # Desired output
    labels = [0, 0, 1, 1]

    # Train Classifer
    clf = tree.DecisionTreeClassifier()  # Decision Tree classifier
    clf = clf.fit(features, labels)  # Find patterns in data

    # Make Predictions
    print (clf.predict([[160, 0]]))
    # Output: 0-apple, 1-orange
    # Correct output is: 1-orange
    outport(locals()) #11 
</py> 

stop 

\ 用 python interpreter 也不是不行,就是功能不多,打字較煩
clf.predict([[100,0]])
clf.predict([[101,0]])
clf.predict([[102,0]])
clf.predict([[103,0]])
clf.predict([[104,0]])
clf.predict([[105,0]])
clf.predict([[106,0]])
clf.predict([[107,0]])

\ 隨便玩玩
    clf :> predict([[120,0]]) tib. 
    clf :> predict([[190,1]]) tib. 

\ 整個分布都打出來看看，好玩！
    101 [for] 101 t@ - 100 + dup . space 
        clf :> predict([[pop(),0]]) tib. [next]
    101 [for] 101 t@ - 100 + dup . space 
        clf :> predict([[pop(),1]]) tib. [next]
    


