'''
    Basic tensorflow matrix  multiply
'''

import tensorflow as tf

matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],
                       [2]])
                       
product = tf.matmul(matrix1, matrix2)  # matrix mltiply like np.dot(m1,m2)

# method 1

sess = tf.Session()
result = sess.run(product)
print('Method 1', result)  # --> [[12]]
sess.close()

# method 2

with tf.Session() as sess:
    result = sess.run(product)
    print('Method 2', result)  # --> [[12]]

