# MINIST手写数字识别实践

MINIST数据集是手写数字的数据库，有60000个训练样本集和10000个测试样本集，每张图片都是28*28像素的灰白图片。

我训练了三种模型来识别手写数字，即逻辑回归、全连接神经网络、卷积神经网络

<br>

## 1.逻辑回归模型

代码请参考：**lab1/logistic regression.py**

模型结构图：

![model](https://github.com/Xumengqi0201/Introduction-to-AI/blob/master/lab1/logistic-regression.png?raw=true)

用MBGD的方法更新W和b，激活函数为sigmoid函数，经过softmax，输出层y0-y9分别代表了数字为0-9的概率，取概率最高的数字所在的下标就是该模型的预测值。

训练集上的准确率达到90%的时候保存模型。之后，加载训练好的模型在测试集上进行测试。

![model](https://github.com/Xumengqi0201/Introduction-to-AI/blob/master/lab1/logistic_log.PNG?raw=true)

模型代码:

```python
# 模型权重
	W = tf.Variable(tf.zeros([784, 10]), name="W")
	b = tf.Variable(tf.zeros([10]), name="b")

	# 用softmax构建逻辑回归模型
	pred = tf.nn.softmax(tf.matmul(x, W) + b)

	# 损失函数(交叉熵)
	cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), 1))

	# 最多训练25个epoch，每次训练一百张图片
	total_epoch = 50
	batch_size = 100   #梯度下降方法为MBGD
	total_batch = int(mnist.train.num_examples / batch_size)

	# 梯度下降
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
```
