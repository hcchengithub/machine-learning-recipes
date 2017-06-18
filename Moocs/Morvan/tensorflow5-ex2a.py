'''
    I want to know if NN can *remembery* the past.
    If x_data is random integer, say 1 5 32 6 87 54 4 23 5 7 78 ....
    And y_data is : 0 0 1 5 32 6 87 54 4 23 5 7 78 .... simply copy from the past of x_data
    with a certain shift.
    [ ] I don't know how to do this yet. 17:38 2017/06/18
'''

import tensorflow as tf
import numpy as np

# create data
x_data = np.random.rand(101).astype(np.int8)
y_data = np.array([0, x_data])

### create tensorflow structure start ###
Weight = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
bias = tf.Variable(tf.zeros([1]))

y = Weight*x_data + bias

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

### create tensorflow structure end ###

sess = tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step % 20 == 0 :
        print(step, sess.run(Weight), sess.run(bias))
        
