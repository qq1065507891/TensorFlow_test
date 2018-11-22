import tensorflow as tf
import os


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bais_varaiable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


def my_conv2d(x, w):
    return tf.nn.conv2d(x, w, filter=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')


def my_pooling(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def activate_func(x, b):
    return tf.nn.relu(x + b)


def net_work(x):
    # 卷积层参数
    w_1 = weight_variable([5, 5, 1, 64])
    b_1 = bais_varaiable([64])
    w_2 = weight_variable([5, 5, 64, 128])
    b_2 = bais_varaiable([128])
    w_3 = weight_variable([5, 5, 128, 256])
    b_3 = bais_varaiable([256])

    # 全连接层参数
    wc_1 = weight_variable([7*7*256, 1000])
    bc_1 = bais_varaiable([1000])
    wc_2 = weight_variable([1000, 10 + 26 + 26])
    bc_2 = bais_varaiable([10 + 26 + 26])


    # 网络结构
    conv_1 = my_conv2d(x, w_1)
    acti_1 = activate_func(conv_1, b_1)
    pool_1 = my_pooling(acti_1)

    conv_2 = my_conv2d(pool_1, w_2) + b_2
    pool_2 = my_pooling(conv_2)

    conv_3 = my_conv2d(x, w_3)
    acti_3 = activate_func(conv_3, b_3)
    pool_3 = my_pooling(acti_3)

    fc_1 = tf.add(tf.multiply(pool_3, wc_1), bc_1)

    fc_2 = tf.add(tf.multiply(fc_1, wc_2), bc_2)
    return fc_2