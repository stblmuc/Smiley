import tensorflow as tf

# create definition of Softmax regression model: prediction = softmax(input * Weights + bias)
def regression(x, nCategories):
	with tf.variable_scope('Regression'):
		W = tf.Variable(tf.zeros([784, nCategories]), name="W")  # weights
		b = tf.Variable(tf.zeros([nCategories]), name="b")  # biases
		y = tf.nn.softmax(tf.matmul(x, W) + b)  # softmax function

	return y, [W, b]