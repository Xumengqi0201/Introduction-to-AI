import tensorflow as tf
import numpy as np

#在当前路径下新建文件夹 MNIST_data MNIST_data MNIST_data，并从 Git上下载并解压
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#划定测试集和训练集
train_images = mnist.train.images
train_labels = mnist.train.labels
test_images = mnist.test.images
test_labels = mnist.test.labels

#占位符，图片像素为28*28=784，输出为10分类
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# 模型权重
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 用softmax构建逻辑回归模型
pred = tf.nn.softmax(tf.matmul(x, W) + b)

# 损失函数(交叉熵)
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), 1))

#最多训练25个epoch，每次训练一百张图片
total_epoch=50
batch_size=100
total_batch = int(mnist.train.num_examples / batch_size)

# 梯度下降
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

saver = tf.train.Saver()
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for epoch in range(total_epoch):

        for i in range(total_batch):
            #每次取100张图片，更新w和b
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            feed_dict_batch={x:batch_xs , y:batch_ys}
            sess.run(optimizer, feed_dict=feed_dict_batch)

        #每个epoch结束以后计算在训练集上的准确率
        correct = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        feed_dict={x:mnist.train.images , y:mnist.train.labels}
        acc = accuracy.eval(feed_dict=feed_dict)
        print('Epoch: %04d, acc=%.7f' % (epoch + 1, acc))

        if(acc>0.9):
            saver.save(sess, "./logistic.model")
            print("Train Complete")
            break

 # 测试模型
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "./logistic.model")#加载参数
    # 测试模型
    correct = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    feed_dict_train = {x: mnist.train.images, y: mnist.train.labels}
    feed_dict_test = {x: mnist.test.images, y: mnist.test.labels}

    acc_train = accuracy.eval(feed_dict=feed_dict_train)
    acc_test = accuracy.eval(feed_dict=feed_dict_test)
    print('train-accuracy=%.7f' % (acc_train))
    print('test-accuracy=%.7f' % (acc_test))

