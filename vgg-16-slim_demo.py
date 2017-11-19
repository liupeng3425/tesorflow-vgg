# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from kits import utils

ROOT_PATH = os.path.dirname(__file__)
PATH = os.path.join(ROOT_PATH, 'cifar-10-batches-py')
LOG_PATH = os.path.join(ROOT_PATH, 'log')

BATCH_SIZE = 64
slim = tf.contrib.slim
vgg = tf.contrib.slim.nets.vgg

train_log_dir = LOG_PATH
if not tf.gfile.Exists(train_log_dir):
    tf.gfile.MakeDirs(train_log_dir)

with tf.Graph().as_default():
    # Set up the data loading:
    data_set = utils.read_data(PATH)
    images, labels = data_set.data_set['data'], data_set.data_set['labels_one_hot']

    # Define the model:
    predictions = vgg.vgg_16(images, is_training=True)

    # Specify the loss function:
    slim.losses.softmax_cross_entropy(predictions, labels)

    total_loss = slim.losses.get_total_loss()
    tf.summary.scalar('losses/total_loss', total_loss)

    # Specify the optimization scheme:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=.001)

    # create_train_op that ensures that when we evaluate it to get the loss,
    # the update_ops are done and the gradient updates are computed.
    train_tensor = slim.learning.create_train_op(total_loss, optimizer)

    # Actually runs training.
    slim.learning.train(train_tensor, train_log_dir)
