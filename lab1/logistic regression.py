import tensorflow as tf
import numpy as np

# 在当前路径下新建文件夹 MNIST_data MNIST_data MNIST_data，并从 Git上下载并解压
import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# 划定测试集和训练集
train_images = mnist.train.images
train_labels = mnist.train.labels
test_images = mnist.test.images
test_labels = mnist.test.labels


def train_model():
	# 占位符，图片像素为28*28=784，输出为10分类
	x = tf.placeholder(tf.float32, [None, 784], name="input_x")
	y = tf.placeholder(tf.float32, [None, 10], name="input_y")

	# 模型权重
	W = tf.Variable(tf.zeros([784, 10]), name="W")
	b = tf.Variable(tf.zeros([10]), name="b")

	# 用softmax构建逻辑回归模型
	pred = tf.nn.softmax(tf.matmul(x, W) + b)

	# 损失函数(交叉熵)
	cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), 1))

	# 最多训练25个epoch，每次训练一百张图片
	total_epoch = 50
	batch_size = 100
	total_batch = int(mnist.train.num_examples / batch_size)

	# 梯度下降
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

	saver = tf.train.Saver()
	tf.add_to_collection('pred_network', pred)
	with tf.Session() as sess:
		tf.global_variables_initializer().run()
		for epoch in range(total_epoch):

			for i in range(total_batch):
				# 每次取100张图片，更新w和b
				batch_xs, batch_ys = mnist.train.next_batch(batch_size)
				feed_dict_batch = {x: batch_xs, y: batch_ys}
				sess.run(optimizer, feed_dict=feed_dict_batch)

			# 每个epoch结束以后计算在训练集上的准确率
			#每一张图片预测值（向量）中最大数的下标，即对应的识别的数字
			pred_train = tf.argmax( sess.run(pred,feed_dict={x:train_images}), 1)
			correct = tf.equal(pred_train,tf.argmax(train_labels,1))
			acc= sess.run( tf.reduce_mean( tf.cast(correct,tf.float32) ))

			print('Epoch: %04d, acc=%.7f' % (epoch + 1, acc))

			if (acc > 0.9):
				saver.save(sess, "./logistic.model")
				print("Train Complete")
				break

def test_model():

	with tf.Session() as sess:
		saver = tf.train.import_meta_graph('./logistic.model.meta')
		saver.restore(sess, "./logistic.model")  # 加载参数

		pred = tf.get_collection('pred_network')[0]
		graph = tf.get_default_graph()
		input_x = graph.get_operation_by_name('input_x').outputs[0]
		pred_test = sess.run(pred, feed_dict={input_x: test_images})
        
		correct = tf.equal(tf.argmax(pred_test, 1), tf.argmax(test_labels,1))
		acc = sess.run( tf.reduce_mean(tf.cast(correct, tf.float32)) )
		print("test accuracy=%.7f" % (acc))

if __name__ == '__main__':

	train_model()#测试时将此行注释掉
	test_model()
