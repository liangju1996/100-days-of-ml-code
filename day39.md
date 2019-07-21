
## Day39代码



### 导入库
[tensorflow](https://blog.51cto.com/zero01/2065598):TensorFlow™是一个基于数据流编程（dataflow programming）的符号数学系统，被广泛应用于各类机器学习（machine learning）算法的编程实现  
[tensorflow中文手册](http://www.tensorfly.cn/tfdoc/tutorials/overview.html)  
[keras](https://keras.io/zh/):Keras 是一个用 Python 编写的高级神经网络 API，它能够以 TensorFlow, CNTK, 或者 Theano 作为后端运行。Keras 的开发重点是支持快速的实验。能够以最小的时延把你的想法转换为实验结果，是做好研究的关键。  
[tensorflow.keras](https://blog.csdn.net/u014061630/article/details/81086564)

```python
import tensorflow.keras as keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# keras自带的MNIST数据集加载方法。MNIST，是一个收录了许多 28 x 28 像素手写数字图片（以灰度值矩阵存储）及其对应的数字的数据集。
# 数组下标以0开始，x_train[0]为输出第一个数
# print(x_train[0])

# %matplotlib inline
plt.imshow(x_train[0], cmap=plt.cm.binary)
# plt.imshow()函数负责对图像进行处理，并显示其格式，但是不能显示。其后跟着plt.show()才能显示出来
#plt.show()
#print(y_train[0])

x_train = tf.keras.utils.normalize(x_train, axis=1)
# 对数据进行归一化处理，1表示横轴，方向从左到右；0表示纵轴，方向从上到下。当axis=1时，数组的变化是横向的，而体现出来的是列的增加或者减少。
x_test = tf.keras.utils.normalize(x_test, axis=1)

#print(x_train[0])
plt.imshow(x_train[0], cmap=plt.cm.binary)
#plt.show()

# 创建模型
model = tf.keras.models.Sequential()
# 输入图像大小为28*28
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# 用relu函数作为激活函数
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# softmax之后输出10个值，分别表示对应的概率
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))
# keras model.compile(loss='目标函数 ', optimizer='adam', metrics=['accuracy'])
# 用来配置模型，编译用来配置模型的学习过程。
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# 模型训练参数设置+训练
model.fit(x_train, y_train, epochs=3)

val_loss, val_acc = model.evaluate(x_test,y_test)
# 预测给定输入的输出，然后计算model.compile中指定的metrics函数，并基于y_true和y_pred，并返回计算的度量值作为输出
#print(val_loss)
#print(val_acc)

# 实际预测，其输出是目标值，根据输入数据预测
predictions = model.predict(x_test)
#print(predictions)
#argmax(a, axis=None, out=None),返回axis维度的最大值的索引
#print(np.argmax(predictions[0]))

plt.imshow(x_test[0], cmap=plt.cm.binary)
#plt.show()
# 保存模型
model.save('epic_num_reader.model')
#加载保存的模型
new_model = tf.keras.models.load_model('epic_num_reader.model')
#测试保存的模型
predictions = new_model.predict(x_test)
print(np.argmax(predictions[0]))
```
