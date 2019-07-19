import tensorflow as tf


# old_v = tf.logging.get_verbosity()
# tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.examples.tutorials.mnist import input_data
# tf.logging.set_verbosity(old_v)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#查看训练数据的大小
print(mnist.train.images.shape)    #(55000, 784)
print(mnist.train.labels.shape)    #(55000, 10)
#查看验证数据的大小
print(mnist.validation.images.shape)
print(mnist.validation.labels.shape)
#查看测试数据的大小
print(mnist.test.images.shape)
print(mnist.test.labels.shape)


x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))

b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = \
    tf. reduce_mean(-tf.reduce_sum(y_* tf.log(y)))
#利用梯度下降法针对模型参数（w，b）进行优化
train_step = tf.train.GradientDescentOptimizer(0.01).minimize
(cross_entropy)


#读入数据
mnist = input_data.read_data_sets ("MNIST_data/", one_hot=True)
# x为训练图像的占位符，y_为训练图像标签的占位符
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

#将单张图片从784维向量重新还原为28*28的矩阵图片
x_image = tf.reshape(x, [-1, 28, 28, 1])

#第一层卷积代码
def weight_variable(shape):
    initial = tf. truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf. constant (0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[l, 1, 1, l], padding＝ 'SAME'）

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[l, 2, 2, l] , strides=[l, 2, 2, 1, padding＝'SAME')

#第一卷积层
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#第二层卷积层
W_conv2 = weight variable([5, 5, 32, 64])
b_conv2 = b工as variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)



#把 1024 维的向量转换成 10 维，对应 10 个类别
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias variable([10])
y_conv = tf.matmul(h_fcl_drop, W_fc2) + b_fc2

#不采用先Softmax再计算交叉熵的方法
#而是用tf.nn.softmax_cross_entropy_with_logits直接计算
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
#同样定义train_step
train_step = tf.train.AdamOptirnizer(1e-4).rninirnize(cross_entropy)

#  定义测试的准确率
correctyrediction = tf.equal(tf.argrnax(y_conv, 1), tf.argrnax(y_, 1))
accuracy= tf.reduce_rnean(tf.cast(correct_prediction, tf.float32))

#创建Session，对变量初始化
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
#训练20000步
for i in range(20000):
    batch= mnist.train.next_batch(50)
    #每100步报告－次在验证集上的准确性
if i % 100 == 0:
    train_accuracy = accuracy.eval(feed_dict={x: batch[O], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g”%（i, train_accuracy)) "
train_step.run(feed_dict={x: batch[O], y_: batch[1], keep_prob: 0.5})

#报告准确性
print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))