import os
from PIL import Image
import numpy as np
import tensorflow as tf
from net import net_work

# 不同字符的数量
CHAR_SET_LEN = 10 + 26 + 26
# 图片高度
IMAGE_HEIGHT = 60
# 图片宽度
IMAGE_WIDTH = 160
# 批次
BATCH_SIZE = 25
# tfrecord 文件存放路径
TFRECORD_FILE = ""

x = tf.placeholder(tf.float32, [None, 28, 28])
y0 = tf.placeholder(tf.float32, [None])
y1 = tf.placeholder(tf.float32, [None])
y2 = tf.placeholder(tf.float32, [None])
y3 = tf.placeholder(tf.float32, [None])

lr = tf.Variable(0.003, dtype=tf.float32)

# 从tfreord读出数据
def read_and_decode(filename):
    # 根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    # 返回文件名和文件
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features={
        'image': tf.FixedLenFeature([], tf.string),
        'label0': tf.FixedLenFeature([], tf.int64),
        'label1': tf.FixedLenFeature([], tf.int64),
        'label2': tf.FixedLenFeature([], tf.int64),
        'label3': tf.FixedLenFeature([], tf.int64),
    })
    # 获取图片数据
    image = tf.decode_raw(features['image'], tf.uint8)
    # tf.train.shuffle_batch必须确定shape
    image = tf.reshape(image, [28, 28])
    # 图片预处理
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    # 获取 label
    label0 = tf.cast(features['label0'], tf.int64)
    label1 = tf.cast(features['label1'], tf.int64)
    label2 = tf.cast(features['label2'], tf.int64)
    label3 = tf.cast(features['label3'], tf.int64)

    return image, label0, label1, label2, label3

# 获取图片数据和标签
image, label0, label1, label2, label3 = read_and_decode(TFRECORD_FILE)

# 使用shuffle_batch可以随机打乱
image_batch, label_label0, label_label1, label_label2, label_lable3 = tf.train.shuffle_batch(
    [image, label0, label1, label2, label3], batch_size=BATCH_SIZE, capacity=50000, min_after_dequeue=10000,num_threads=1
)

# 定义网络结构
"""暂时未定义网络结构"""

with tf.Session() as sess:
    #  inputs: a tensor of size [batch_size, height, width, channels]
    X = tf.reshape(x, [BATCH_SIZE, 28, 28, 1])
    # 数据输入网络得到输出值
    logist0, logist1, logist2, logist3 = net_work(X)
    one_hot_labes0 = tf.one_hot(indices=tf.cast(y0, tf.int32), depth=CHAR_SET_LEN)
    one_hot_labes1 = tf.one_hot(indices=tf.cast(y1, tf.int32), depth=CHAR_SET_LEN)
    one_hot_labes2 = tf.one_hot(indices=tf.cast(y2, tf.int32), depth=CHAR_SET_LEN)
    one_hot_labes3 = tf.one_hot(indices=tf.cast(y3, tf.int32), depth=CHAR_SET_LEN)

    # 计算loss
    loss0 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logists=logist0, lables=one_hot_labes0))
    loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logists=logist1, lables=one_hot_labes1))
    loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logists=logist2, lables=one_hot_labes2))
    loss3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logists=logist3, lables=one_hot_labes3))
    # 计算总的loss
    total_loss = (loss0+loss1+loss2+loss3) / 4
    # 优化total_loss
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(total_loss)

    # 计算准确率
    corecet_predintion0  = tf.equal(tf.argmax(one_hot_labes0, 1), tf.argmax(logist0, 1))
    accuracy0 = tf.reduce_mean(tf.cast(corecet_predintion0), tf.float32)

    corecet_predintion1 = tf.equal(tf.argmax(one_hot_labes1, 1), tf.argmax(logist1, 1))
    accuracy1 = tf.reduce_mean(tf.cast(corecet_predintion1), tf.float32)

    corecet_predintion2 = tf.equal(tf.argmax(one_hot_labes2, 1), tf.argmax(logist2, 1))
    accuracy2 = tf.reduce_mean(tf.cast(corecet_predintion2), tf.float32)

    corecet_predintion3 = tf.equal(tf.argmax(one_hot_labes3, 1), tf.argmax(logist3, 1))
    accuracy3 = tf.reduce_mean(tf.cast(corecet_predintion3), tf.float32)

    # 创建一个协调器， 管理线程
    coord = tf.train.Coordinator()
    # 启动QueueRunner， 此时文件名队列已经进队
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    saver = tf.train.Saver(max_to_keep=5)

    while True:
        i= 0
        # 获取一个批次的数据和标签
        b_image, b_label0, b_label1, b_label2, b_label3 = sess.run([image_batch, label_label0, label_label1, label_label2, label_lable3])
        # 优化模型
        sess.run(optimizer, feed_dict={x:b_image, y0:b_label0, y1:b_label1, y2:b_label2, y3:b_label3})

        # 每次迭代100次计算一次loss和准确率
        if i % 100 == 0:
            # 每迭代20000次降低一次学习率
            if i % 20000 == 0:
                sess.run(tf.assign(lr, lr / 3))
            acc0, acc1, acc2, acc3, loss_ = sess.run([accuracy0, accuracy1,accuracy2,accuracy3, total_loss],feed_dict={
                x: b_image, y0: b_label0, y1: b_label1, y2: b_label2, y3: b_label3
            } )
            learing_rate = sess.run(lr)
            print('Iter:%d Loss:%d Accuracy:%.2f, %.2f, %.2f, %.2f  Learning_rate:%.4f' % (i, loss_, acc0, acc1, acc2, acc3, learing_rate))

            if (acc0 + acc1 + acc2 + acc3) / 4 > 0.95:
                saver.save(sess, "./model/captcha.pkl", global_step=i)
                break
            elif i % 2000 == 0:
                saver.save(sess, "./model/captcha.pkl", global_step=i)
        i += 1

    # 通知其他线程关闭
    coord.request_stop()
    # 其他所有线程关闭之后， 这一函数才能返回
    coord.join(threads)
