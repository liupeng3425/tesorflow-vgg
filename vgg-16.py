import logging.config
import os

import tensorflow as tf

from kits import utils

logging.config.fileConfig("./logging.conf")

# create logger
logger_name = "vgg-16-pytorch"
log = logging.getLogger(logger_name)
PATH = os.path.dirname(__file__)
PATH = os.path.join(PATH, 'cifar-10-batches-py')
BATCH_SIZE = 64


def conv(input_img, kernel_size):
    return tf.nn.conv2d(input_img, kernel_size, padding='SAME', strides=[1, 1, 1, 1])


def max_pool(input_img, name):
    return tf.nn.max_pool(input_img, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def weight_variable(name, shape):
    initial = tf.truncated_normal(shape, mean=0, stddev=0.01)
    return tf.Variable(initial, name=name)


def bias_variable(name, shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial, name=name)


block_filter_num = (64, 128, 256, 512, 512)
fc_neural_num = (4096, 4096, 10)
softmax_output_num = 10

num_of_maxpool = 5

data = tf.placeholder(tf.float32, [None, 32, 32, 3], 'input')

w_conv1 = weight_variable('w_conv1', [3, 3, 3, block_filter_num[0]])
b_conv1 = bias_variable('b_conv1', [block_filter_num[0]])
conv1 = tf.nn.relu(conv(data, w_conv1) + b_conv1)

w_conv2 = weight_variable('w_conv2', [3, 3, block_filter_num[0], block_filter_num[0]])
b_conv2 = bias_variable('b_conv2', [block_filter_num[0]])
conv2 = tf.nn.relu(conv(conv1, w_conv2) + b_conv2)

max_pool_2 = max_pool(conv2, 'max_pool_2')

w_conv3 = weight_variable('w_conv3', [3, 3, block_filter_num[0], block_filter_num[1]])
b_conv3 = bias_variable('b_conv3', [block_filter_num[1]])
conv3 = tf.nn.relu(conv(max_pool_2, w_conv3) + b_conv3)

w_conv4 = weight_variable('w_conv4', [3, 3, block_filter_num[1], block_filter_num[1]])
b_conv4 = bias_variable('b_conv4', [block_filter_num[1]])
conv4 = tf.nn.relu(conv(conv3, w_conv4) + b_conv4)

max_pool_4 = max_pool(conv4, 'max_pool_4')

w_conv5 = weight_variable('w_conv5', [3, 3, block_filter_num[1], block_filter_num[2]])
b_conv5 = bias_variable('b_conv5', [block_filter_num[2]])
conv5 = tf.nn.relu(conv(max_pool_4, w_conv5) + b_conv5)

w_conv6 = weight_variable('w_conv6', [3, 3, block_filter_num[2], block_filter_num[2]])
b_conv6 = bias_variable('b_conv6', [block_filter_num[2]])
conv6 = tf.nn.relu(conv(conv5, w_conv6) + b_conv6)

w_conv7 = weight_variable('w_conv7', [3, 3, block_filter_num[2], block_filter_num[2]])
b_conv7 = bias_variable('b_conv7', [block_filter_num[2]])
conv7 = tf.nn.relu(conv(conv6, w_conv7) + b_conv7)

max_pool_7 = max_pool(conv7, 'max_pool_7')

w_conv8 = weight_variable('w_conv8', [3, 3, block_filter_num[2], block_filter_num[3]])
b_conv8 = bias_variable('b_conv8', [block_filter_num[3]])
conv8 = tf.nn.relu(conv(max_pool_7, w_conv8) + b_conv8)

w_conv9 = weight_variable('w_conv9', [3, 3, block_filter_num[3], block_filter_num[3]])
b_conv9 = bias_variable('b_conv9', [block_filter_num[3]])
conv9 = tf.nn.relu(conv(conv8, w_conv9) + b_conv9)

w_conv10 = weight_variable('w_conv10', [3, 3, block_filter_num[3], block_filter_num[3]])
b_conv10 = bias_variable('b_conv10', [block_filter_num[3]])
conv10 = tf.nn.relu(conv(conv9, w_conv10) + b_conv10)

max_pool_9 = max_pool(conv10, 'max_pool_9')

w_conv11 = weight_variable('w_conv11', [3, 3, block_filter_num[3], block_filter_num[4]])
b_conv11 = bias_variable('b_conv11', [block_filter_num[4]])
conv11 = tf.nn.relu(conv(max_pool_9, w_conv11) + b_conv11)

w_conv12 = weight_variable('w_conv12', [3, 3, block_filter_num[4], block_filter_num[4]])
b_conv12 = bias_variable('b_conv12', [block_filter_num[4]])
conv12 = tf.nn.relu(conv(conv11, w_conv12) + b_conv12)

w_conv13 = weight_variable('w_conv13', [3, 3, block_filter_num[4], block_filter_num[4]])
b_conv13 = bias_variable('b_conv13', [block_filter_num[4]])
conv13 = tf.nn.relu(conv(conv12, w_conv13) + b_conv13)

max_pool_13 = max_pool(conv13, 'max_pool_13')

dim = block_filter_num[4] * data.get_shape()[1].value * data.get_shape()[1].value / (2 ** (2 * num_of_maxpool))
dim = int(dim)
flat = tf.reshape(max_pool_13, [-1, dim])

w_fc14 = weight_variable('w_fc14', [dim, fc_neural_num[0]])
b_fc14 = bias_variable('b_fc14', [fc_neural_num[0]])
fc14 = tf.nn.relu(tf.matmul(flat, w_fc14) + b_fc14)

w_fc15 = weight_variable('w_fc15', [fc_neural_num[0], fc_neural_num[1]])
b_fc15 = bias_variable('b_fc15', [fc_neural_num[1]])
fc15 = tf.nn.relu(tf.matmul(fc14, w_fc15) + b_fc15)

w_fc16 = weight_variable('w_fc16', [fc_neural_num[1], softmax_output_num])
b_fc16 = bias_variable('b_fc16', [softmax_output_num])
fc16 = tf.matmul(fc15, w_fc16) + b_fc16

softmax = tf.nn.softmax(fc16)
y_label = tf.placeholder(tf.float32, [None, softmax_output_num])
cross_entropy = -tf.reduce_mean(y_label * tf.log(softmax))

train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(softmax, 1), tf.argmax(y_label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

data_set = utils.read_data(PATH)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for i in range(40000):
    batch = data_set.next_batch_data(BATCH_SIZE)

    # for var in tf.trainable_variables():
    #     print(var)

    if i % 250 == 0:
        loss, train_accuracy, prediction, fc15_v, fc14_v = \
            sess.run([cross_entropy, accuracy, softmax, conv9, conv6],
                     feed_dict={data: batch['data'],
                                y_label: batch['labels_one_hot']})
        log.info("step %d, training accuracy %g, cross entropy %g" % (i, train_accuracy, loss))
        # print(tf.get_default_graph().get_tensor_by_name('w_conv1:0').eval()[0][0][0][0])
        # print('prediction:')
        # print(prediction[0:2])
        # print('prediction_sum')
        # print(numpy.sum(prediction[0:2], axis=1))
    if i % 500 == 0:
        log.info("test accuracy %g" % accuracy.eval(feed_dict={
            data: data_set.test_set['data'][0:2000], y_label: data_set.test_set['labels_one_hot'][0:2000]}))

    train_step.run(feed_dict={data: batch['data'], y_label: batch['labels_one_hot']})

log.info("test accuracy %g" % accuracy.eval(feed_dict={
    data: data_set.test_set['data'][4000:8000], y_label: data_set.test_set['labels_one_hot'][4000:8000]}))
