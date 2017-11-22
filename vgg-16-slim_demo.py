# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from kits import utils
from tensorflow.contrib.slim import nets

ROOT_PATH = os.path.dirname(__file__)
PATH = os.path.join(ROOT_PATH, 'cifar-10-batches-py')
LOG_PATH = os.path.join(ROOT_PATH, 'log')

BATCH_SIZE = 64
slim = tf.contrib.slim
vgg = nets.vgg

train_log_dir = LOG_PATH
if not tf.gfile.Exists(train_log_dir):
    tf.gfile.MakeDirs(train_log_dir)


def vgg16(inputs):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        net = slim.max_pool2d(net, [2, 2], scope='pool5')
        net = slim.fully_connected(net, 4096, scope='fc6')
        net = slim.dropout(net, 0.5, scope='dropout6')
        net = slim.fully_connected(net, 4096, scope='fc7')
        net = slim.dropout(net, 0.5, scope='dropout7')
        net = slim.fully_connected(net, 10, activation_fn=None, scope='fc8')
    return net


with tf.Graph().as_default():
    # Set up the data loading:
    data_set = utils.read_data(PATH)
    images, labels = data_set.data_set['data'], data_set.data_set['labels_one_hot']

    # Define the model:
    predictions = vgg16(images)
    predictions = tf.reshape(predictions, [-1, 10])

    # Specify the loss function:
    tf.losses.softmax_cross_entropy(labels, predictions)

    total_loss = slim.losses.get_total_loss()
    tf.summary.scalar('losses/total_loss', total_loss)

    # Specify the optimization scheme:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=.001)

    # create_train_op that ensures that when we evaluate it to get the loss,
    # the update_ops are done and the gradient updates are computed.
    train_tensor = slim.learning.create_train_op(total_loss, optimizer)

    # Actually runs training.
    slim.learning.train(train_tensor, train_log_dir)
