import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# 初始化数据集
def init_data_sets():
    return input_data.read_data_sets("E:\ml_service_backup\data\mnist", one_hot=True)


# 获取训练数据
def get_data_sets(data, batch_size):
    return data.train.next_batch(batch_size)
# 获取测试数据
def get_test_sets(data, batch_size):
    return data.test.next_batch(batch_size)


# 构建输入输出
def construct_io():
    x = tf.placeholder('float', [None, 784])
    y = tf.placeholder('float', [None, 10])
    return x, y


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# 计算实际值
def construct_module(x, y):
    # 第一层卷积 滤波器5*5 输入通道数量1 输出通道数量（滤波器个数）32
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    # 把x变成一个4d向量，其第2、第3维对应图片的宽、高，最后一维代表图片的颜色通道数
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # 第二层卷积 滤波器5*5 输入通道数量1 输出通道数量（滤波器个数）32
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # 全连接层
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # dropout 用一个placeholder来代表一个神经元的输出在dropout中保持不变的概率。
    # 这样我们可以在训练过程中启用dropout，在测试过程中关闭dropout。
    # keep_prob = tf.placeholder('float')
    # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    h_fc1_drop = h_fc1

    # 输出层
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    return y_conv


# 计算loss
def calc_loss(y, y_hat):
    return -tf.reduce_sum(y * tf.log(y_hat))


# 测试精度
def calc_accuracy(y, y_hat):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_hat, 1))
    accuracy = tf.cast(correct_prediction, 'float')
    return tf.reduce_mean(accuracy)


import argparse
import os
import sys
import time
import tensorflow as tf


def run_training():
    # 初始化数据集
    data = init_data_sets()

    # 构造模型
    x, y = construct_io()
    y_hat = construct_module(x, y)
    loss = calc_loss(y, y_hat)
    # train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(loss)
    train_step = tf.train.AdadeltaOptimizer(FLAGS.learning_rate).minimize(loss)
    accuracy = calc_accuracy(y, y_hat)

    saver = tf.train.Saver()
    # 初始化变量
    init = tf.global_variables_initializer()
    # 启动图
    with tf.Session() as sess:
        if os.path.exists(FLAGS.variable_file+'.index'):
            # 从文件中恢复变量
            saver.restore(sess, FLAGS.variable_file)
            print("Model restored.")
        else:
            print("Model inited.")
            sess.run(init)


        for step in range(FLAGS.max_steps):
            start_time = time.time()
            batch_xs, batch_ys = get_data_sets(data, FLAGS.batch_size)
            _, loss_value = sess.run([train_step, loss], feed_dict={x: batch_xs, y: batch_ys})
            duration = time.time() - start_time
            if step % 100 == 0:
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                batch_xs, batch_ys = get_test_sets(data, FLAGS.batch_size)
                # accuracy_value = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
                # print('accuracy=%.2f ' % (accuracy_value))
            # 存储变量到文件
            if step % 1000 == 0:
                save_path = saver.save(sess, FLAGS.variable_file)
                print("Model saved in file: ", save_path)
        save_path = saver.save(sess, FLAGS.variable_file)
        print("Model saved in file: ", save_path)


def main(_):
    run_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.0001,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=30000,
        help='Number of steps to run trainer.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=10,
        help='Batch size.  Must divide evenly into the dataset sizes.'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                             'tensorflow/mnist/logs/fully_connected_feed'),
        help='Directory to put the log data.'
    )
    parser.add_argument(
        '--variable_file',
        type=str,
        default='E:\ml_service_backup\data\mnist\\variable.ckpt',
        help='Directory to put the log data.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
