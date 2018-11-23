import os
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 不同字符的数量
CHAR_SET_LEN = 10 + 26 + 26
# 图片高度
IMAGE_HEIGHT = 60
# 图片宽度
IMAGE_WIDTH = 160
# 批次
BATCH_SIZE = 1
# tfrecord 文件存放路径
TFRECORD_FILE = ""

x = tf.placeholder(tf.float32, [None, 28, 28])

# 从tfreord读出数据
def read_and_decode(filename):
    # 根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    # 返回文件名和文件
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features={
        'image': tf.FixedLenFeature([], tf.string),
        'label0': tf.FixedLenFeature([], tf.string),
        'label1': tf.FixedLenFeature([], tf.string),
        'label2': tf.FixedLenFeature([], tf.string),
        'label3': tf.FixedLenFeature([], tf.string),
    })
    # 获取图片数据
    image = tf.decode_raw(features['image'], tf.uint8)
    # tf.train.shuffle_batch必须确定shape
    image = tf.reshape(image, [28, 28])
    imae_raw = tf.reshape(image, [28, 28])
    # 图片预处理
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    # 获取 label
    label0 = tf.cast(features['label0'], tf.int64)
    label1 = tf.cast(features['label1'], tf.int64)
    label2 = tf.cast(features['label2'], tf.int64)
    label3 = tf.cast(features['label3'], tf.int64)

    return image,image_raw ,label0, label1, label2, label3

# 获取图片数据和标签
image, image_raw,label0, label1, label2, label3 = read_and_decode(TFRECORD_FILE)

# 使用shuffle_batch可以随机打乱
image_batch, image_raw_batch,label_label0, label_label1, label_label2, label_lable3 = tf.train.shuffle_batch(
    [image, image_raw, label0, label1, label2, label3], batch_size=BATCH_SIZE, capacity=50000, min_after_dequeue=10000,num_threads=1
)

# 定义网络结构
"""暂时未定义网络结构"""

with tf.Session() as sess:
    #  inputs: a tensor of size [batch_size, height, width, channels]
    X = tf.reshape(x, [BATCH_SIZE, 28, 28, 1])
    # 数据输入网络得到输出值
    logist0, logist1, logist2, logist3 = trian_network_fn(X)
    # 预测值
    prediction0 = tf.reshape(logist0, [-1, CHAR_SET_LEN])
    prediction0 = tf.argmax(prediction0, 1)

    prediction1 = tf.reshape(logist1, [-1, CHAR_SET_LEN])
    prediction1 = tf.argmax(prediction1, 1)

    prediction2 = tf.reshape(logist2, [-1, CHAR_SET_LEN])
    prediction2 = tf.argmax(prediction2, 1)

    prediction3 = tf.reshape(logist3, [-1, CHAR_SET_LEN])
    prediction3 = tf.argmax(prediction3, 1)

    saver = tf.train.Saver()
    saver.restore(sess, './model/captcha.pkl-')

    # 创建一个协调器， 管理线程
    coord = tf.train.Coordinator()
    # 启动QueueRunner， 此时文件名队列已经进队
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(10):
        # 获取一个批次的数据和标签
        b_image, b_image_raw, b_label0, b_label1, b_label2, b_label3 = sess.run([image_batch, image_raw_batch,label_label0, label_label1, label_label2, label_lable3])
        # 优化模型
        img = Image.fromarray(b_image_raw[0], "L")
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        print('label:', b_label0, b_label1, b_label2, b_label3)
        # 预测
        label0, label1, label2, label3 = sess.run([prediction0, prediction1, prediction2, prediction3], feed_dict={x: b_image})
        # 打印预测值
        print('predict:', label0, label1, label2, label3)
    # 通知其他线程关闭
    coord.request_stop()
    # 其他所有线程关闭之后， 这一函数才能返回
    coord.join(threads)
