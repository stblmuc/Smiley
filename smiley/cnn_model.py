from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import configparser
import os

# CNN with standard tensorflow
def convolutional(x, nCategories, is_training=True):
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), 'trainConfig.ini'))
    image_size = int(config['DEFAULT']['IMAGE_SIZE'])

    with tf.variable_scope('cnn'):
        x_image = tf.reshape(x, [-1, image_size, image_size, 1])

        # Convolutional layer + max pool layer
        conv1 = tf.layers.conv2d(x_image, 32, 5, activation=tf.nn.relu)
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # Convolutional layer + max pool layer
        conv2 = tf.layers.conv2d(conv1, 64, 5, activation=tf.nn.relu)
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        flatten = tf.contrib.layers.flatten(conv2)

        # Fully connected layer + dropout
        fc1 = tf.layers.dense(flatten, 1024)
        fc1 = tf.layers.dropout(fc1, rate=0.5, training=is_training)

        # Fully connected layer + softmax
        fc2 = tf.layers.dense(fc1, nCategories)
        out = tf.nn.softmax(fc2)
        
    return out, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="cnn")
