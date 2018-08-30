import configparser
import os
import tensorflow as tf


# create definition of softmax regression model: prediction = softmax(input * Weights + bias)
def regression(x, nCategories):
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), 'config.ini'))
    image_size = int(config['DEFAULT']['IMAGE_SIZE'])

    with tf.variable_scope('regression', reuse=tf.AUTO_REUSE):
        W = tf.Variable(tf.zeros([image_size * image_size, nCategories]), name="W")  # weights
        b = tf.Variable(tf.zeros([nCategories]), name="b")  # biases
        y = tf.nn.softmax(tf.matmul(x, W) + b)  # softmax function

    return y, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="regression")
