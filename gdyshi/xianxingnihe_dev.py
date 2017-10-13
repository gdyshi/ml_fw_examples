import argparse
import os
import sys
import time
import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets import base


# 初始化数据集
def init_data_sets():
    x_data = np.linspace(-100, 100, 10000)
    y_data = x_data * 0.2 + 0.3 + np.random.normal(0, 0.55, x_data.size)
    return base.Datasets(train=x_data, validation=y_data, test=x_data)


# 获取训练数据
def get_data_sets(data, batch_size):
  return data.train, data.validation


# 计算y_hat
def construct_module(x, y):
    W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    y = W * x + b
    return y


# 计算loss
def calc_loss(y, y_hat):
  return tf.reduce_mean(tf.square(y - y_hat))


def run_training():
  # 初始化数据集
  data = init_data_sets()

  # 构造模型
  x = tf.placeholder('float', [None])
  y = tf.placeholder('float', [None])
  y_hat = construct_module(x, y)
  loss = calc_loss(y, y_hat)
  train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(loss)
  # 初始化变量
  init = tf.global_variables_initializer()
  # 启动图
  with tf.Session() as sess:
      sess.run(init)

      for step in range(FLAGS.max_steps):
          start_time = time.time()
          batch_xs, batch_ys = get_data_sets(data, FLAGS.batch_size)
          _, loss_value = sess.run([train_step, loss], feed_dict={x: batch_xs, y: batch_ys})
          duration = time.time() - start_time
          if step % 100 == 0:
              print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))


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
      default=2000,
      help='Number of steps to run trainer.'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      help='Batch size.  Must divide evenly into the dataset sizes.'
  )
  parser.add_argument(
      '--log_dir',
      type=str,
      default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           'tensorflow/mnist/logs/fully_connected_feed'),
      help='Directory to put the log data.'
  )

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
