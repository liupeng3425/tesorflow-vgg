# -*- coding: utf-8 -*-
from tensorflow.contrib import slim
import tensorflow as tf
import os
from kits import utils

ROOT_PATH = os.path.dirname(__file__)
PATH = os.path.join(ROOT_PATH, 'cifar-10-batches-py')
LOG_PATH = os.path.join(ROOT_PATH, 'log')
BATCH_SIZE = 64


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


data_set = utils.read_data(PATH)
data = tf.placeholder(tf.float32, [None, 32, 32, 3], 'input')
y_label = tf.placeholder(tf.float32, [None, 10])

vgg = vgg16(data)
vgg = tf.reshape(vgg, [-1, 10])
cross_entropy = slim.losses.softmax_cross_entropy(vgg, y_label)
train_step = tf.train.AdamOptimizer(1e-2).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(vgg, 1), tf.argmax(y_label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for i in range(40000):
    batch = data_set.next_batch_data(BATCH_SIZE)

    if i % 250 == 0:
        loss, train_accuracy, prediction = sess.run([cross_entropy, accuracy, vgg],
                                                    feed_dict={data: batch['data'],
                                                               y_label: batch['labels_one_hot']})
        print("step %d, training accuracy %g, cross entropy %g" % (i, train_accuracy, loss))
        # print(tf.get_default_graph().get_tensor_by_name('w_conv1:0').eval()[0][0][0][0])
        # print('prediction:')
        # print(prediction[0:2])
        # print('prediction_sum')
        # print(numpy.sum(prediction[0:2], axis=1))
    if i % 500 == 0:
        print("test accuracy %g" % accuracy.eval(feed_dict={
            data: data_set.test_set['data'][0:2000], y_label: data_set.test_set['labels_one_hot'][0:2000]}))

    train_step.run(feed_dict={data: batch['data'], y_label: batch['labels_one_hot']})

print("test accuracy %g" % accuracy.eval(feed_dict={
    data: data_set.test_set['data'][4000:8000], y_label: data_set.test_set['labels_one_hot'][4000:8000]}))
