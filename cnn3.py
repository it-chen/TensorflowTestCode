from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

IMAGE_SIZE = 224
FLAGS = tf.app.flags.FLAGS
# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 1000
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1281167
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 50000
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")

def my_conv_net(images, train=False):
    # conv1_1
    with tf.variable_scope('conv1_1') as scope:
        conv1_1_weight = tf.Variable(tf.truncated_normal([3, 3, 1, 32],
                                                       stddev=0.1, dtype=tf.float32))
        conv1_1 = tf.nn.conv2d(images, conv1_1_weight, [1, 1, 1, 1], padding='SAME')
        conv1_1_bias = tf.Variable(tf.zeros([32], dtype=tf.float32))
        conv1_1 = tf.nn.relu(tf.nn.bias_add(conv1_1, conv1_1_bias), name=scope.name)

    # conv1_2
    with tf.variable_scope('conv1_2') as scope:

        conv1_2_weight = tf.Variable(tf.truncated_normal([3, 3, 32, 32],
                                                       stddev=0.1, dtype=tf.float32))
        conv1_2 = tf.nn.conv2d(conv1_1, conv1_2_weight, [1, 1, 1, 1], padding='SAME')
        conv1_2_bias = tf.Variable(tf.zeros([32], dtype=tf.float32))
        conv1_2 = tf.nn.relu(tf.nn.bias_add(conv1_2, conv1_2_bias), name=scope.name)

    # pool1
    pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    # conv2_1
    with tf.variable_scope('conv2_1') as scope:
        # kernel = _variable_with_weight_decay('weights', shape=[3, 3, 32, 64], wd=0.000, layer_name=scope.name)
        # conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        # biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.0), layer_name=scope.name)
        # conv2_1 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope.name)
        # _activation_summary(conv2_1)

        conv2_1_weight = tf.Variable(tf.truncated_normal([3, 3, 32, 64],
                                                       stddev=0.1, dtype=tf.float32))
        conv2_1 = tf.nn.conv2d(pool1, conv2_1_weight, [1, 1, 1, 1], padding='SAME')
        conv2_1_bias = tf.Variable(tf.zeros([64], dtype=tf.float32))
        conv2_1 = tf.nn.relu(tf.nn.bias_add(conv2_1, conv2_1_bias), name=scope.name)

    # conv2_2
    with tf.variable_scope('conv2_2') as scope:
        # kernel = _variable_with_weight_decay('weights', shape=[3, 3, 128, 128], wd=0.000, layer_name=scope.name)
        # conv = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
        # biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.0), layer_name=scope.name)
        # conv2_2 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope.name)
        # _activation_summary(conv2_2)
        conv2_2_weight = tf.Variable(tf.truncated_normal([3, 3, 64, 64],
                                                       stddev=0.1, dtype=tf.float32))
        conv2_2 = tf.nn.conv2d(conv2_1, conv2_2_weight, [1, 1, 1, 1], padding='SAME')
        conv2_2_bias = tf.Variable(tf.zeros([64], dtype=tf.float32))
        conv2_2 = tf.nn.relu(tf.nn.bias_add(conv2_2, conv2_2_bias), name=scope.name)


    # pool2
    pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # fc6
    with tf.variable_scope('fc6') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=[dim, 4096], wd=0.000, layer_name=scope.name)
        biases = _variable_on_cpu('biases', [4096], tf.constant_initializer(0.0), layer_name=scope.name)
        fc6 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        if train:
            fc6 = tf.nn.dropout(fc6, 0.5)
        # _activation_summary(fc6)

    # fc7
    with tf.variable_scope('fc7') as scope:
        weights = _variable_with_weight_decay('weights', shape=[4096, 4096], wd=0.000, layer_name=scope.name)
        biases = _variable_on_cpu('biases', [4096], tf.constant_initializer(0.0), layer_name=scope.name)
        fc7 = tf.nn.relu(tf.matmul(fc6, weights) + biases, name=scope.name)
        if train:
            fc7 = tf.nn.dropout(fc7, 0.5)
        # _activation_summary(fc7)

    # linear layer(WX + b),
    # We don't apply softmax here because
    # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
    # and performs the softmax internally for efficiency.
    with tf.variable_scope('softmax_linear') as scope:
        # weights = _variable_with_weight_decay('weights', [4096, 20], wd=0.000, layer_name='fc8')
        # biases = _variable_on_cpu('biases', [20], tf.constant_initializer(0.0), layer_name='fc8')

        # weights = tf.Variable(tf.truncated_normal([4096, 20],
        #                                                stddev=0.1, dtype=tf.float32))
        # biases = tf.Variable(tf.zeros([20], dtype=tf.float32))
        weights = _variable_with_weight_decay('weights', shape=[4096, 20], wd=0.000, layer_name=scope.name)
        biases = _variable_on_cpu('biases', [20], tf.constant_initializer(0.0), layer_name=scope.name)
        softmax_linear = tf.add(tf.matmul(fc7, weights), biases, name=scope.name)
    #     _activation_summary(softmax_linear)

    return softmax_linear

def _variable_with_weight_decay(name, shape, wd, layer_name):
    var = _variable_on_cpu(
        name,
        shape,
        tf.contrib.layers.xavier_initializer(uniform=False),
        layer_name)
    # if wd is not None:
    #     weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    #     tf.add_to_collection('losses', weight_decay)
    return var

def _variable_on_cpu(name, shape, initializer, layer_name):
    with tf.device('/cpu:0'):
        # dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        dtype = tf.float32
        # if FLAGS.use_pre_trained:
        #     if name == 'weights':
        #         filename = FLAGS.weights_dir + layer_name + '_W.npy'
        #     else:  # biases
        #         filename = FLAGS.weights_dir + layer_name + '_b.npy'
        #     init = tf.constant(np.load(filename), dtype=tf.float32)
        #     var = tf.get_variable(name, initializer=init, dtype=dtype)
        # else:
        init = initializer
        var = tf.get_variable(name, shape, initializer=init, dtype=dtype)
    return var