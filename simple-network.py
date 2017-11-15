import os

import tensorflow as tf

from kits import utils

PATH = os.path.dirname(__file__)
PATH = os.path.join(PATH, 'cifar-10-batches-py')
BATCH_SIZE = 64

def conv(input_img, kernel_size):
    return tf.nn.conv2d(input_img, kernel_size, padding='SAME', strides=[1, 1, 1, 1])


def max_pool(input_img, name):
    return tf.nn.max_pool(input_img, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def weight_variable(name, shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(name, shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


data = tf.placeholder(tf.float32, (None, 32, 32, 3))

w_conv1 = weight_variable('w_conv1', (3, 3, 3, 64))
b_conv1 = bias_variable('b_conv1', [64])
conv1 = tf.nn.relu(conv(data, w_conv1) + b_conv1)

max_pool1 = max_pool(conv1, 'max_pool1')

w_conv2 = weight_variable('w_conv2', (3, 3, 64, 64))
b_conv2 = bias_variable('b_conv2', [64])
conv2 = tf.nn.relu(conv(max_pool1, w_conv2) + b_conv2)

max_pool2 = max_pool(conv2, 'max_pool2')

flat = tf.reshape(max_pool2, (-1, 4096))
dim = flat.get_shape()[1].value
w_fc3 = weight_variable('w_fc3', (dim, 1024))
b_fc3 = bias_variable('b_fc3', [1024])
fc3 = tf.nn.relu(tf.matmul(flat, w_fc3) + b_fc3)

w_softmax = weight_variable('w_softmax', (1024, 10))
b_softmax = bias_variable('b_softmax', [10])
# softmax = tf.nn.softmax(tf.matmul(fc3, w_softmax) + b_softmax)
softmax = tf.matmul(fc3, w_softmax) + b_softmax

y_label = tf.placeholder(tf.float32, (None, 10))
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_label, logits=softmax))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(softmax, 1), tf.argmax(y_label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

data_set = utils.read_data(PATH)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for i in range(10000):
    batch = data_set.next_batch_data(BATCH_SIZE)
    sess.run(train_step, feed_dict={data: batch['data'], y_label: batch['labels_one_hot']})
    # for var in tf.trainable_variables():
    #     print(var)

    if i % 500 == 0:
        loss, train_accuracy, prediction = sess.run([cross_entropy, accuracy, softmax],
                                                    feed_dict={data: batch['data'],
                                                               y_label: batch['labels_one_hot']})
        print("step %d, training accuracy %g, cross entropy %g" % (i, train_accuracy, loss))
        print("test accuracy %g" % accuracy.eval(feed_dict={
            data: data_set.test_set['data'], y_label: data_set.test_set['labels_one_hot']}))

print("test accuracy %g" % accuracy.eval(feed_dict={
    data: data_set.test_set['data'], y_label: data_set.test_set['labels_one_hot']}))
