'''
	学会用 Tensorflow 自带的 tensorboard 去可视化我们所建造出来的神经网络

	Published on Jun 20, 2016
	python3 简单教学教程
	本节练习代码: https://github.com/MorvanZhou/tutoria...
	学会用 Tensorflow 自带的 tensorboard 去可视化我们所建造出来的神经网络
	是一个很好的学习理解方式.用最直观的流程图告诉你你的神经网络是长怎样,有
	助于你发现编程中间的问题和疑问.

	播放列表 Play list: https://www.youtube.com/playlist?list=PLXO45tsB95cKI5AIlf5TxxFPzb-0zeVZ8

	我创建的学习网站: http://morvanzhou.github.io/tutorials/
	微博更新: 莫烦python
	QQ 机器学习讨论群: 531670665
	Category
	Science & Technology
	License
	Standard YouTube License

    10:39 2017-06-19
'''
import pdb
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 這樣就是神經網路的一層了
def add_layer(inputs, in_size, out_size, activation_function=None):
	# Add one more layer and return the output of this layer
    # Matrix 大寫
	with tf.name_scope('layer'):  # 我想這個名字 layer 是 TensorBoard 認得且預期的
		with tf.name_scope('weights'):
			Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')  

		# bias is an array, not matrix, 小寫
		with tf.name_scope('biases'):
			biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')  # 使不為零   

		with tf.name_scope('Wx_plus_b'):
			Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)  # 這是神經元的前級
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

with tf.name_scope('inputs'):  # 我想這個名字 inputs 是 TensorBoard 認得且預期的
	xs = tf.placeholder(tf.float32, [None, 1], name='x_input')  # 1 input
	ys = tf.placeholder(tf.float32, [None, 1], name='y_input')  # 1 output

# 這一題一層 3 個神經元就夠了! 100 個不見得好。
# 多加幾層也不見更好
l1 = add_layer(xs, 1, 9, activation_function=tf.nn.relu)
# l2 = add_layer(l1, 9, 9, activation_function=tf.nn.relu)
# l3 = add_layer(l2, 9, 9, activation_function=tf.nn.relu)
prediction = add_layer(l1, 9, 1, activation_function=None)

with tf.name_scope('loss'):
	# 每個 tf.method(..., name="blabla") 都可以給個名字
	loss = tf.reduce_mean(
				tf.reduce_sum(
					tf.square(ys - prediction, name="square"), reduction_indices=[1],
					name="reduce_sum"
				),
				name="reduce_mean"
			)

with tf.name_scope('train'):
	train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss) # Learning Rate < 1

	
sess = tf.Session()

# 把所有 tensorboard 需要的訊息都放進 logs/ 以便投向 web browser 顯示
# tf.train.SummaryWriter soon be deprecated, use following
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:  # tensorflow version < 0.12
    writer = tf.train.SummaryWriter('logs/', sess.graph)
else: # tensorflow version >= 0.12
    writer = tf.summary.FileWriter("logs/", sess.graph)

# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()

sess.run(init)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()

for i in range(1000):
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
1.	Run 過本程式之後，產生 logs/ 裡面的東西。
2.	執行下面的 tensorboard web server 就可以上網頁看了
	G:\Morvan>tensorboard --logdir="logs"
	Starting TensorBoard b'47' at http://0.0.0.0:6006
	(Press CTRL+C to quit)
3.	Windows 10 上執行要用這個網址： http://127.0.0.1:6006/
4.	到了該網頁，要選 GRAPHS tab 裡面才有東西。
	真的就是那個圖！！！！
	
'''
    
