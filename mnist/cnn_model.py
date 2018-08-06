from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib.slim as slim


# Create definition of CNN model with slim API
def CNN(inputs, nCategories=2, is_training=True):
    batch_norm_params = {'is_training': is_training, 'decay': 0.9, 'updates_collections': None}
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        x = tf.reshape(inputs, [-1, 28, 28, 1])  # input layer: reshape image to 28x28 pixel matrix

        # For slim.conv2d, default argument values are:
        # padding='SAME', activation_fn=nn.relu,
        # weights_initializer = initializers.xavier_initializer(),
        # biases_initializer = init_ops.zeros_initializer
        # stride (int or [int, int]) = 1
        # slim.conv2d(inputs, num_outputs, kernel_size (int or [int, int]), scope)

        # slim.max_pool2d defaults: stride=2, padding='VALID',
        # slim.max_pool2d(inputs: `[batch_size, height, width, channels]`, kernel_size (int or [int, int]), scope)

        net = slim.conv2d(x, 32, [5, 5], scope='conv1')  # 1st layer
        net = slim.max_pool2d(net, [2, 2], scope='pool1')  # max-pooling step
        net = slim.conv2d(net, 64, [5, 5], scope='conv2')  # 2nd layer
        net = slim.max_pool2d(net, [2, 2], scope='pool2')  # max-pooling step
        net = slim.flatten(net, scope='flatten3')  # flattening step

        # For slim.fully_connected, default argument values are:
        # activation_fn = nn.relu,
        # weights_initializer = initializers.xavier_initializer(),
        # biases_initializer = init_ops.zeros_initializer
        net = slim.fully_connected(net, 1024, scope='fc3')  # 3rd layer
        net = slim.dropout(net, is_training=is_training, scope='dropout3')  # dropout: 0.5 by default
        outputs = slim.fully_connected(net, nCategories, activation_fn=tf.nn.softmax, normalizer_fn=None,
                                       scope='fco')  # output layer

    return outputs, slim.get_model_variables()

# CNN with normal tensorflow
def convolutional(x, nCategories=2, is_training=True):

    with tf.variable_scope('cnn'):

        x_image = tf.reshape(x, [-1, 28, 28, 1])

        conv1 = tf.layers.conv2d(x_image, 32, 5, activation=tf.nn.relu)
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        conv2 = tf.layers.conv2d(conv1, 64, 5, activation=tf.nn.relu)
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        fc1 = tf.contrib.layers.flatten(conv2)

        fc1 = tf.layers.dense(fc1, 1024)

        fc1 = tf.layers.dropout(fc1, rate=0.5, training=is_training)

        out = tf.layers.dense(fc1, nCategories)
    
        y = tf.nn.softmax(out)
        
    return y, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="cnn") 