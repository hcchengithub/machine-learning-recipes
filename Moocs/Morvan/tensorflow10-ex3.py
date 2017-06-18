'''
    TensorFlow 10 Example 3, Add layers function

'''
import pdb
import tensorflow as tf


# 這樣就是神經網路的一層了
def add_layer(input, in_size, out_size, activation_function=None)
    # Matrix 大寫
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))  
    # bias is an array, not matrix, 小寫
    biases = tf.Variable(tf.zeros([1, out_size])) + 0.1)  # 使不為零   
    Wx_plus_b = tf.matmul(input, Weights) + biases  # 這是神經元的前級
    if activation_function is None:
        # 沒有 activation function 就是直通, 線性關係。
        outpts = Wx_plus_b 
    else:
        outpts = activation_function(Wx_plus_b)
    return outputs
    
