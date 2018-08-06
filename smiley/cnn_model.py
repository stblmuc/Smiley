from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

# CNN with standard tensorflow
def convolutional(x, nCategories=2, is_training=True):
    with tf.variable_scope('cnn'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])

        conv1 = tf.layers.conv2d(x_image, 32, 5, activation=tf.nn.relu)
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        conv2 = tf.layers.conv2d(conv1, 64, 5, activation=tf.nn.relu)
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        flatten = tf.contrib.layers.flatten(conv2)

        fc1 = tf.layers.dense(flatten, 1024)
        fc1 = tf.layers.dropout(fc1, rate=0.5, training=is_training)

        fc2 = tf.layers.dense(fc1, nCategories)
        out = tf.nn.softmax(fc2)
        
    return out, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="cnn")
