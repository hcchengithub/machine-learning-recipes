'''
    TensorFlow Placeholder

'''
import pdb
import tensorflow as tf

input1 = tf.placeholder(tf.float32) 
input2 = tf.placeholder(tf.float32) 

output = tf.multiply(input1, input2)  # tf.ml() 要改成 tf.multiply()

with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1:[7], input2:[2]}))
