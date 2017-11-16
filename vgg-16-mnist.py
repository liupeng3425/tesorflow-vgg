import os

import numpy
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

PATH = os.path.dirname(__file__)
PATH = os.path.join(PATH, 'MNIST_data')
BATCH_SIZE = 64


def conv(input_img, kernel_size):
    return tf.nn.conv2d(input_img, kernel_size, padding='SAME', strides=[1, 1, 1, 1])


def max_pool(input_img, name):
    return tf.nn.max_pool(input_img, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def weight_variable(name, shape):
    initial = tf.random_normal(shape, stddev=0.001)
    return tf.Variable(initial, name=name)


def bias_variable(name, shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial, name=name)


block_filter_num = (64, 128, 128, 32, 32)
fc_neural_num = (128, 128, 64)
softmax_output_num = 10

num_of_maxpool = 1

data = tf.placeholder(tf.float32, [None, 28 * 28 * 1], 'input')
reshape = tf.reshape(data, (-1, 28, 28, 1))

conv1 = tf.nn.relu(
    conv(reshape, weight_variable('w_conv1', [3, 3, 1, block_filter_num[0]])) +
    bias_variable('b_conv1', [block_filter_num[0]]))
conv2 = tf.nn.relu(
    conv(conv1, weight_variable('w_conv2', [3, 3, block_filter_num[0], block_filter_num[0]])) +
    bias_variable('b_conv2', [block_filter_num[0]]))

# max_pool_2 = max_pool(conv2, 'max_pool_2')

conv3 = tf.nn.relu(conv(conv2, weight_variable('w_conv3', [3, 3, block_filter_num[0], block_filter_num[1]])) +
                   bias_variable('b_conv3', [block_filter_num[1]]))
conv4 = tf.nn.relu(conv(conv3, weight_variable('w_conv4', [3, 3, block_filter_num[1], block_filter_num[1]])) +
                   bias_variable('b_conv4', [block_filter_num[1]]))

# max_pool_4 = max_pool(conv4, 'max_pool_4')

conv5 = tf.nn.relu(conv(conv4, weight_variable('w_conv5', [3, 3, block_filter_num[1], block_filter_num[2]])) +
                   bias_variable('b_conv5', [block_filter_num[2]]))
conv6 = tf.nn.relu(conv(conv5, weight_variable('w_conv6', [3, 3, block_filter_num[2], block_filter_num[2]])) +
                   bias_variable('b_conv6', [block_filter_num[2]]))
conv7 = tf.nn.relu(conv(conv6, weight_variable('w_conv7', [3, 3, block_filter_num[2], block_filter_num[2]])) +
                   bias_variable('b_conv7', [block_filter_num[2]]))

max_pool_7 = max_pool(conv7, 'max_pool_7')

conv8 = tf.nn.relu(conv(max_pool_7, weight_variable('w_conv8', [3, 3, block_filter_num[2], block_filter_num[3]])) +
                   bias_variable('b_conv8', [block_filter_num[3]]))
conv9 = tf.nn.relu(conv(conv8, weight_variable('w_conv9', [3, 3, block_filter_num[3], block_filter_num[3]])) +
                   bias_variable('b_conv9', [block_filter_num[3]]))
conv10 = tf.nn.relu(conv(conv9, weight_variable('w_conv10', [3, 3, block_filter_num[3], block_filter_num[3]])) +
                    bias_variable('b_conv10', [block_filter_num[3]]))

# max_pool_9 = max_pool(conv10, 'max_pool_9')

conv11 = tf.nn.relu(conv(conv10, weight_variable('w_conv11', [3, 3, block_filter_num[3], block_filter_num[4]])) +
                    bias_variable('b_conv11', [block_filter_num[4]]))
conv12 = tf.nn.relu(conv(conv11, weight_variable('w_conv12', [3, 3, block_filter_num[4], block_filter_num[4]])) +
                    bias_variable('b_conv12', [block_filter_num[4]]))
conv13 = tf.nn.relu(conv(conv12, weight_variable('w_conv13', [3, 3, block_filter_num[4], block_filter_num[4]])) +
                    bias_variable('b_conv13', [block_filter_num[4]]))

# max_pool_13 = max_pool(conv13, 'max_pool_13')

dim = block_filter_num[4] * reshape.get_shape()[1].value * reshape.get_shape()[1].value / (2 ** (2 * num_of_maxpool))
dim = int(dim)
flat = tf.reshape(conv13, [-1, dim])
fc14 = tf.nn.relu(tf.matmul(flat, weight_variable('w_fc14', [dim, fc_neural_num[0]])) +
                  bias_variable('b_fc14', [fc_neural_num[0]]))

fc15 = tf.nn.relu(tf.matmul(fc14, weight_variable('w_fc15', [fc_neural_num[0], fc_neural_num[1]])) +
                  bias_variable('b_fc15', [fc_neural_num[1]]))

fc16 = tf.nn.relu(tf.matmul(fc15, weight_variable('w_fc16', [fc_neural_num[1], fc_neural_num[2]])) +
                  bias_variable('b_fc16', [fc_neural_num[2]]))

softmax = tf.matmul(fc16, weight_variable('w_softmax', [fc_neural_num[2], softmax_output_num])) + \
          bias_variable('b_softmax', [softmax_output_num])
softmax = tf.nn.softmax(softmax)
# softmax = tf.matmul(fc16, gen_variable('w_softmax', [1000, 10])) + gen_variable('b_softmax', 10)

y_label = tf.placeholder(tf.float32, [None, softmax_output_num])
cross_entropy = -tf.reduce_sum(y_label * tf.log(softmax))
train_step = tf.train.AdamOptimizer(1e-2).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(softmax, 1), tf.argmax(y_label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

data_set = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for i in range(40000):
    batch = data_set.train.next_batch(BATCH_SIZE)

    # for var in tf.trainable_variables():
    #     print(var)

    if i % 250 == 0:
        loss, train_accuracy, prediction = sess.run([cross_entropy, accuracy, softmax],
                                                    feed_dict={data: batch[0],
                                                               y_label: batch[1]})
        print("step %d, training accuracy %g, cross entropy %g" % (i, train_accuracy, loss))
        # print(tf.get_default_graph().get_tensor_by_name('w_conv1:0').eval()[0][0][0][0])
        print('prediction:')
        print(prediction[0:2])
        print('prediction_sum')
        print(numpy.sum(prediction[0:2], axis=1))
    if i % 500 == 0:
        print("test accuracy %g" % accuracy.eval(feed_dict={
            data: data_set.test.images[1000:2000], y_label: data_set.test.labels[1000:2000]}))

    train_step.run(feed_dict={data: batch[0], y_label: batch[1]})

print("test accuracy %g" % accuracy.eval(feed_dict={
    data: data_set.test.images[1000:3000], y_label: data_set.test.labels[1000:3000]}))
