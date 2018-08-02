import tensorflow as tf


# create definition of Softmax regression model
def regression(x, categories=2):
    W = tf.Variable(tf.zeros([784, categories]), name="W")  # weights
    b = tf.Variable(tf.zeros([categories]), name="b")  # biases
    y = tf.nn.softmax(tf.matmul(x, W) + b)  # softmax function
    return y, [W, b]
