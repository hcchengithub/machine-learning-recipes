'''
    TensorFlow 10 Example 3, Add layers function
    21:32 2017/06/18
'''
import pdb
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 這樣就是神經網路的一層了
def add_layer(input, in_size, out_size, activation_function=None):
    # Matrix 大寫
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))  
    # bias is an array, not matrix, 小寫
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)  # 使不為零   
    Wx_plus_b = tf.matmul(input, Weights) + biases  # 這是神經元的前級
    if activation_function is None:
        # 沒有 activation function 就是直通, 線性關係。
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# 開始安排實驗, 讓神經網路做擬合
x_data = np.linspace(-1, 1, 300)[:,np.newaxis]  # np.newaxis 幹嘛的? 看下面實驗, 看不懂。
noise = np.random.normal(0, 0.05, x_data.shape)  # 方差 0.05, 跟 x_data 一樣的 shape 
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32, [None, 1])  # 1 input
ys = tf.placeholder(tf.float32, [None, 1])  # 1 output

# 這一題一層 3 個神經元就夠了! 100 個不見得好。
# 多加幾層也不見更好
l1 = add_layer(xs, 1, 9, activation_function=tf.nn.relu)
# l2 = add_layer(l1, 9, 9, activation_function=tf.nn.relu)
# l3 = add_layer(l2, 9, 9, activation_function=tf.nn.relu)
prediction = add_layer(l1, 9, 1, activation_function=None)

loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.square(ys - prediction), reduction_indices=[1]
            )
        )

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss) # Learning Rate < 1

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()

for i in range(10000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        # print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
        plt.pause(0.5)

'''
# 一層 10 個神經元的執行結果 
C:\  Users\hcche\AppData\Local\Programs\Python\Python36\python.exe C:/Users/hcche/Documents/GitHub/machine-learning-recipes/Moocs/Morvan/tensorflow10-ex3.py
WARNING:tensorflow:From C:/Users/hcche/Documents/GitHub/machine-learning-recipes/Moocs/Morvan/tensorflow10-ex3.py:42: initialize_all_variables (from tensorflow.python.ops.variables) is deprecated and will be removed after 2017-03-02.
Instructions for updating:
Use `tf.global_variables_initializer` instead.
2017-06-18 21:29:20.927197: W d:\build\tensorflow\tensorflow-r1.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE instructions, but these are available on your machine and could speed up CPU computations.
2017-06-18 21:29:20.927508: W d:\build\tensorflow\tensorflow-r1.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE2 instructions, but these are available on your machine and could speed up CPU computations.
2017-06-18 21:29:20.927801: W d:\build\tensorflow\tensorflow-r1.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE3 instructions, but these are available on your machine and could speed up CPU computations.
2017-06-18 21:29:20.928061: W d:\build\tensorflow\tensorflow-r1.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2017-06-18 21:29:20.928342: W d:\build\tensorflow\tensorflow-r1.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-06-18 21:29:20.928630: W d:\build\tensorflow\tensorflow-r1.1\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
0.541812
0.00669779
0.00561613
0.00496033
0.00440168
0.00404401
0.00377204
0.00359453
0.00346263
0.00336211
0.00328647
0.0032164
0.00315045
0.00309457
0.00303905
0.00299452
0.00295583
0.00292147
0.00288845
0.00285853     loss 收斂到一個程度

Process finished with exit code 0


# np.newaxis 幹嘛的? 看下面實驗, 看不懂。
np.linspace(-1, 1, 3)
array([-1.,  0.,  1.])
np.linspace(-1, 1, 30)
array([-1.        , -0.93103448, -0.86206897, -0.79310345, -0.72413793,
       -0.65517241, -0.5862069 , -0.51724138, -0.44827586, -0.37931034,
       -0.31034483, -0.24137931, -0.17241379, -0.10344828, -0.03448276,
        0.03448276,  0.10344828,  0.17241379,  0.24137931,  0.31034483,
        0.37931034,  0.44827586,  0.51724138,  0.5862069 ,  0.65517241,
        0.72413793,  0.79310345,  0.86206897,  0.93103448,  1.        ])
np.linspace(-1, 1, 30)[:, np.newaxis]
array([[-1.        ],
       [-0.93103448],
       [-0.86206897],
       [-0.79310345],
       [-0.72413793],
       [-0.65517241],
       [-0.5862069 ],
       [-0.51724138],
       [-0.44827586],
       [-0.37931034],
       [-0.31034483],
       [-0.24137931],
       [-0.17241379],
       [-0.10344828],
       [-0.03448276],
       [ 0.03448276],
       [ 0.10344828],
       [ 0.17241379],
       [ 0.24137931],
       [ 0.31034483],
       [ 0.37931034],
       [ 0.44827586],
       [ 0.51724138],
       [ 0.5862069 ],
       [ 0.65517241],
       [ 0.72413793],
       [ 0.79310345],
       [ 0.86206897],
       [ 0.93103448],
       [ 1.        ]])
np.linspace(-1, 1, 30)[:, 3]
Traceback (most recent call last):
  File "<input>", line 1, in <module>
IndexError: too many indices for array
np.linspace(-1, 1, 3)[:, 3]
Traceback (most recent call last):
  File "<input>", line 1, in <module>
IndexError: too many indices for array
np.linspace(-1, 1, 3)[:, np.newaxis]
array([[-1.],
       [ 0.],
       [ 1.]])
np.linspace(-1, 1, 3)[:,]
array([-1.,  0.,  1.])
np.linspace(-1, 1, 3)
array([-1.,  0.,  1.])
np.linspace(-1, 1, 3)[:,]
array([-1.,  0.,  1.])
np.linspace(-1, 1, 3)[:]
array([-1.,  0.,  1.])

'''
    
