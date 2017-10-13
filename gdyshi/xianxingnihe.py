import tensorflow as tf
import numpy as np

# 生成数据
x_data = np.linspace(-100, 100, 10000)
y_data = x_data * 0.2 + 0.3 + np.random.normal(0, 0.55, x_data.size)

# 创建一个变量, 初始化为标量 0.
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name='W')
# 创建一个变量, 初始化为标量 0.2
b = tf.Variable(tf.constant(0.2), name='b')

y = W * x_data + b
loss = tf.reduce_mean(tf.square(y - y_data), name='loss')
optimizer = tf.train.GradientDescentOptimizer(0.00001)
train = optimizer.minimize(loss)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
print("W=", sess.run(W), "b=", sess.run(b), "loss=", sess.run(loss))

for step in range(20000):
    sess.run(train)
    print("W=", sess.run(W), "b=", sess.run(b), "loss=", sess.run(loss))
