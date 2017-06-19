#!/usr/bin/env python
# from __future__ import print_function
import pdb
import tensorflow as tf
import cv2
import sys

sys.path.append("game/")
import tetris_fun as game
import random
import numpy as np
from collections import deque
import PIL.Image as image


def save_screen(bitmap):
	row, col, color = bitmap.shape
	pic_new = image.new("RGB", (row, col))	# "L" 灰階, "RGB" 彩色
	for i in range(row):
		for j in range(col):
			pic_new.putpixel((i, j), tuple(bitmap[i, j]))
	pic_new.save("screen.jpg", "JPEG")


def save_80x80(bitmap):
	row, col = bitmap.shape
	pic_new = image.new("L", (row, col))  # "L" 灰階, "RGB" 彩色
	for i in range(row):
		for j in range(col):
			pic_new.putpixel((i, j), int(bitmap[i, j]))
	pic_new.save("80x80.jpg", "JPEG")

GAME = 'tetris'
ACTIONS = 5
GAMMA = 0.99
OBSERVE = 10000.
EXPLORE = 3000000.
FINAL_EPSILON = 0.0001
<<<<<<< HEAD
INITIAL_EPSILON = 0.5
=======
INITIAL_EPSILON = 0.001
>>>>>>> 11d3cb44789f28d68646fce14e58aaf663dbb9af
REPLAY_MEMORY = 50000
BATCH = 32
FRAME_PER_ACTION = 1


# pdb.set_trace() # breakpoint 1
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.01)
	return tf.Variable(initial)


def bias_variable(shape):
	initial = tf.constant(0.01, shape=shape)
	return tf.Variable(initial)


# Convolution and Max pooling, so this is CNN
# See 27:36 @ 「一天搞懂深度學習--學習心得-ZrEsLwCjdxY.mp4」
def conv2d(x, W, stride):  # convolution
	return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")


def max_pool_2x2(x):  # Max pooling
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def createNetwork():
	# 定义深度神经网络的参数和偏置
	W_conv1 = weight_variable([8, 8, 4, 32])
	b_conv1 = bias_variable([32])

	W_conv2 = weight_variable([4, 4, 32, 64])
	b_conv2 = bias_variable([64])

	W_conv3 = weight_variable([3, 3, 64, 64])
	b_conv3 = bias_variable([64])

	W_fc1 = weight_variable([1600, 512])
	b_fc1 = bias_variable([512])

	W_fc2 = weight_variable([512, ACTIONS])
	b_fc2 = bias_variable([ACTIONS])

	# 输入层
	s = tf.placeholder("float", [None, 80, 80, 4])

	# 隐藏层
	h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)

	h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)

	h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

	h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)
	# RELU see 19:00 in 「一天搞懂深度學習--學習心得-ZrEsLwCjdxY.mp4」

	# 输出层
	readout = tf.matmul(h_fc1, W_fc2) + b_fc2

	return s, readout, h_fc1


def trainNetwork(s, readout, h_fc1, sess):
	# s 是 input layer, readout 是 output layer
	# 定义损失函数
	a = tf.placeholder("float", [None, ACTIONS])
	y = tf.placeholder("float", [None])
	readout_action = tf.reduce_sum(tf.multiply(readout, a),
								   reduction_indices=1)
	cost = tf.reduce_mean(tf.square(y - readout_action))
	train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

	# 开启游戏模拟器，会打开一个模拟器的窗口，实时显示游戏的信息
	game_state = game.GameState()
	# pdb.set_trace()  # 布置電腦螢幕畫面
	# 创建双端队列用于存放replay memory
	D = deque()

	# 获取游戏的初始状态，设置动作为不执行跳跃，并将初始状态修改成80*80*4大小
	do_nothing = np.zeros(ACTIONS)
	do_nothing[0] = 1
	x_t, r_0, terminal = game_state.frame_step(do_nothing)	# 對 game 做動作
	x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
	ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
	s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

	# 用于加载或保存网络参数
	saver = tf.train.Saver()
	sess.run(tf.initialize_all_variables())
	checkpoint = tf.train.get_checkpoint_state("saved_networks")  # path of saved network
	# pdb.set_trace() # breakpoint
	if checkpoint and checkpoint.model_checkpoint_path:
		# pdb.set_trace()  # 接力
		# (Pdb) checkpoint.model_checkpoint_path --> 'saved_networks\\bird-dqn-10000'
		saver.restore(sess, checkpoint.model_checkpoint_path)
		print("Successfully loaded:", checkpoint.model_checkpoint_path)
	else:
		# pdb.set_trace()  # 從頭開始
		print("Could not find old network weights")

	# 开始训练
	epsilon = INITIAL_EPSILON
	t = 0
	while "flappy bird" != "angry bird":
		# pdb.set_trace() # breakpoint
		# 使用epsilon贪心策略选择一个动作
		readout_t = readout.eval(feed_dict={s: [s_t]})[0]
		a_t = np.zeros([ACTIONS])  # 要餵給 NN 都用這種最終形式的編碼
		action_index = 0
		if t % FRAME_PER_ACTION == 0:
			# 执行一个随机动作
			if random.random() <= epsilon:
				print("----------Random Action----------")
				action_index = random.randrange(ACTIONS)  # 0..4 random integer
				a_t[random.randrange(ACTIONS)] = 1
			# 由神经网络计算的Q(s,a)值选择对应的动作
			else:
				action_index = np.argmax(readout_t)
				a_t[action_index] = 1
		else:
			a_t[0] = 1	# 不执行跳跃动作
			pdb.set_trace() # 這行永遠到不了吧？

		# 随游戏的进行，不断降低epsilon，减少随机动作
		if epsilon > FINAL_EPSILON and t > OBSERVE:
			epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

		# 执行选择的动作，并获得下一状态及回报
		x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
		x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)),
							cv2.COLOR_BGR2GRAY)
		ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
<<<<<<< HEAD
		#'''
		# 大作业2 俄罗斯方块 ─ 畫面經「池化」等加工簡化過頭了嗎？ 
		if t % 47 == 0: 
			pdb.set_trace()
			pass
		#	save_screen(x_t1_colored)
		#	save_80x80(x_t1)
		# 以上抓圖查看證實，似乎不會。
		#'''
=======
		# 大作业2 俄罗斯方块 ─ 畫面經「池化」等加工簡化過頭了嗎？ 
		# if t % 47 == 0: 
		# 	pdb.set_trace()
		# 	save_screen(x_t1_colored)
		# 	save_80x80(x_t1)
		# 以上抓圖查看證實，似乎不會。
>>>>>>> 11d3cb44789f28d68646fce14e58aaf663dbb9af
		x_t1 = np.reshape(x_t1, (80, 80, 1))
		s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

		# 将状态转移过程存储到D中，用于更新参数时采样
		D.append((s_t, a_t, r_t, s_t1, terminal))
		if len(D) > REPLAY_MEMORY:
			D.popleft()

		# 过了观察期，才会进行网络参数的更新
		if t > OBSERVE:
			# 从D中随机采样，用于参数更新
			minibatch = random.sample(D, BATCH)

			# 分别将当前状态、采取的动作、获得的回报、下一状态分组存放
			s_j_batch = [d[0] for d in minibatch]
			a_batch = [d[1] for d in minibatch]
			r_batch = [d[2] for d in minibatch]
			s_j1_batch = [d[3] for d in minibatch]

			# 计算Q(s,a)的新值
			y_batch = []
			readout_j1_batch = readout.eval(feed_dict={s: s_j1_batch})
			for i in range(0, len(minibatch)):
				terminal = minibatch[i][4]
				# 如果游戏结束，则只有反馈值
				if terminal:
					y_batch.append(r_batch[i])
				else:
					y_batch.append(r_batch[i] +
								   GAMMA * np.max(readout_j1_batch[i]))

			# 使用梯度下降更新网络参数
			train_step.run(feed_dict={
				y: y_batch,
				a: a_batch,
				s: s_j_batch}
			)

		# 状态发生改变，用于下次循环
		s_t = s_t1
		t += 1

		# 每进行10000次迭代，保留一下网络参数
		if t % 10000 == 0:
			saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step=t)

		# 打印游戏信息
		state = ""
		if t <= OBSERVE:
			state = "observe"
		elif OBSERVE < t <= OBSERVE + EXPLORE:
			state = "explore"
		else:
			state = "train"

		if terminal:
			gover = 1
		else:	
			gover = 0
		print("TIMESTEP", t, "/ STATE", state, \
<<<<<<< HEAD
			  # "/ EPSILON", epsilon, \
			  # "/ ACTION", action_index, \
			  # "/ Terminal", gover, \
			  # "/ Score", game_state.score, \
=======
			  "/ EPSILON", epsilon, \
			  "/ ACTION", action_index, \
			  "/ Terminal", gover, \
			  "/ Score", game_state.score, \
>>>>>>> 11d3cb44789f28d68646fce14e58aaf663dbb9af
			  "/ REWARD", r_t, \
			  )	 # "/ Q_MAX %e" % np.max(readout_t))
		# if r_t != 0 :
		# 	pdb.set_trace()  # check reward 


def playGame():
	sess = tf.InteractiveSession()
	s, readout, h_fc1 = createNetwork()
	trainNetwork(s, readout, h_fc1, sess)


if __name__ == "__main__":
	playGame()