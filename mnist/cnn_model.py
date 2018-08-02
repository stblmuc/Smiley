from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib.slim as slim


# Create definition of CNN model with slim API
def CNN(inputs, categories=2, is_training=True):
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
        outputs = slim.fully_connected(net, categories, activation_fn=None, normalizer_fn=None,
                                       scope='fco')  # output layer

    return outputs, slim.get_model_variables()
