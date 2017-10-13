import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 初始化数据集
def init_data_sets():
    return input_data.read_data_sets("E:\ml_service_backup\data\mnist", one_hot=True)


# 获取训练数据
def get_data_sets(data, batch_size):
    return data.train.next_batch(batch_size)


# 计算实际值
def construct_module(x, y):
    W = tf.Variable(tf.random_uniform([784, 10], -1.0, 1.0))
    b = tf.Variable(tf.random_uniform([10], -1.0, 1.0))
    return tf.nn.softmax(tf.matmul(x, W) + b)


# 构建输入输出
def construct_io():
    x = tf.placeholder('float', [None, 784])
    y = tf.placeholder('float', [None, 10])
    return x, y


# 计算loss
def calc_loss(y, y_hat):
    return -tf.reduce_sum(y * tf.log(y_hat))


# 测试精度
def calc_accuracy(y, y_hat):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_hat, 1))
    accuracy = tf.cast(correct_prediction, 'float')
    return tf.reduce_mean(accuracy)
