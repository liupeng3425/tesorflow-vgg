import numpy
import tensorflow as tf
import scipy
from kits import utils
import matplotlib.pyplot as plt
import os

PATH = os.path.dirname(__file__)
PATH = os.path.join(PATH, 'cifar-10-batches-py')
BATCH_SIZE = 64


def conv(input_img, kernel_size):
    return tf.nn.conv2d(input_img, kernel_size, padding='SAME', strides=[1, 1, 1, 1])


def max_pool(input_img, name):
    return tf.nn.max_pool(input_img, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def gen_variable(name, shape):
    return tf.get_variable(name=name,
                           shape=shape,
                           dtype=numpy.float32,
                           initializer=tf.random_uniform_initializer)


data = tf.placeholder(numpy.float32, [BATCH_SIZE, 32, 32, 3], 'input')

conv1 = tf.nn.relu(conv(data, gen_variable('w_conv1', [3, 3, 3, 64])) + gen_variable('b_conv1', [64]))
conv2 = tf.nn.relu(conv(conv1, gen_variable('w_conv2', [3, 3, 64, 64])) + gen_variable('b_conv2', [64]))

max_pool_2 = max_pool(conv2, 'max_pool_2')

conv3 = tf.nn.relu(conv(max_pool_2, gen_variable('w_conv3', [3, 3, 64, 128])) + gen_variable('b_conv3', [128]))
conv4 = tf.nn.relu(conv(conv3, gen_variable('w_conv4', [3, 3, 128, 128])) + gen_variable('b_conv4', [128]))

# max_pool_4 = max_pool(conv4, 'max_pool_4')

conv5 = tf.nn.relu(conv(conv4, gen_variable('w_conv5', [3, 3, 128, 256])) + gen_variable('b_conv5', [256]))
conv6 = tf.nn.relu(conv(conv5, gen_variable('w_conv6', [3, 3, 256, 256])) + gen_variable('b_conv6', [256]))
conv7 = tf.nn.relu(conv(conv6, gen_variable('w_conv7', [3, 3, 256, 256])) + gen_variable('b_conv7', [256]))

max_pool_7 = max_pool(conv7, 'max_pool_7')

conv8 = tf.nn.relu(conv(max_pool_7, gen_variable('w_conv8', [3, 3, 256, 512])) + gen_variable('b_conv8', [512]))
conv9 = tf.nn.relu(conv(conv8, gen_variable('w_conv9', [3, 3, 512, 512])) + gen_variable('b_conv9', [512]))
conv10 = tf.nn.relu(conv(conv9, gen_variable('w_conv10', [3, 3, 512, 512])) + gen_variable('b_conv10', [512]))

# max_pool_9 = max_pool(conv10, 'max_pool_9')

conv11 = tf.nn.relu(conv(conv10, gen_variable('w_conv11', [3, 3, 512, 512])) + gen_variable('b_conv11', [512]))
conv12 = tf.nn.relu(conv(conv11, gen_variable('w_conv12', [3, 3, 512, 512])) + gen_variable('b_conv12', [512]))
conv13 = tf.nn.relu(conv(conv12, gen_variable('w_conv13', [3, 3, 512, 512])) + gen_variable('b_conv13', [512]))

max_pool_9 = max_pool(conv9, 'max_pool_9')

flat = tf.reshape(max_pool_9, [64, -1])
dim = flat.get_shape()[1].value
fc14 = tf.nn.relu(tf.matmul(flat, gen_variable('w_fc14', [dim, 4096])) +
                  gen_variable('b_fc14', [4096]))

fc15 = tf.nn.relu(tf.matmul(fc14, gen_variable('w_fc15', [4096, 4096])) +
                  gen_variable('b_fc15', [4096]))

fc16 = tf.nn.relu(tf.matmul(fc15, gen_variable('w_fc16', [4096, 1000])) +
                  gen_variable('b_fc16', [1000]))

softmax = tf.nn.softmax(tf.matmul(fc16, gen_variable('w_softmax', [1000, 10])) + gen_variable('b_softmax', 10))

y_label = tf.placeholder(numpy.float32, [64, 1])
cross_entropy = -tf.reduce_sum(y_label * tf.log(softmax))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(softmax, 1), tf.argmax(y_label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

data_set = utils.read_data(PATH)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for i in range(10000):
    batch = data_set.next_batch_data(64)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            data: batch['data'], y_label: batch['labels']})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={data: batch['data'], y_label: batch['labels']})

print("test accuracy %g" % accuracy.eval(feed_dict={
    data: data_set.test_set['data'], y_label: data_set.test_set['labels']}))
