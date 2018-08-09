import tensorflow as tf
import configparser
import os

# create definition of Softmax regression model: prediction = softmax(input * Weights + bias)
def regression(x, categories):
	config = configparser.ConfigParser()
	config.read(os.path.join(os.path.dirname(__file__), 'trainConfig.ini'))
	image_size = int(config['DEFAULT']['IMAGE_SIZE'])

	with tf.variable_scope('regression'):
		W = tf.Variable(tf.zeros([image_size * image_size, categories]), name="W")  # weights
		b = tf.Variable(tf.zeros([categories]), name="b")  # biases
		y = tf.nn.softmax(tf.matmul(x, W) + b)  # softmax function

	return y, [W, b]