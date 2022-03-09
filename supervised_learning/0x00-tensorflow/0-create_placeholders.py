import numpy as np
import tensorflow as tf
def create_placeholders(nx, classes):
    pass

a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0) # also tf.float32 implicitly
total = a + b
print(a)
print(b)
print(total)

sess = tf.Session()
print(sess.run(total))
