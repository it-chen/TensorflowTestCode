import tensorflow as tf
a=tf.constant([[3., 3.]])
b=tf.constant([[2.],[2.]])
product = tf.matmul(a,b)
sess = tf.Session()
result = sess.run(product)
print(result)