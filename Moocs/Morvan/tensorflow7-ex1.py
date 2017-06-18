'''
    TensorFlow Variable

'''
import pdb
import tensorflow as tf

state = tf.Variable(0, name='counter')
pdb.set_trace()  # print(state.name)
one = tf.constant(1)

new_value = tf.add(state, one)  #  it is a function, I think.
update = tf.assign(state, new_value)  # it is a fnction too

init = tf.initialize_all_variables()  # must have if define variable

with tf.Session() as sess:
    sess.run(init)
    for i in range(3):
        sess.run(update)
        print(sess.run(state))
