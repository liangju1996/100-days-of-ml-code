




### 代码
```python
import tensorflow.keras as keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 数组下标以0开始，x_train[0]为输出第一个数
# print(x_train[0])

# %matplotlib inline
plt.imshow(x_train[0], cmap=plt.cm.binary)
#plt.show()
#print(y_train[0])

x_train = tf.keras.utils.normalize(x_train, axis=1)
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
# 配置模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# 模型训练参数设置+训练
model.fit(x_train, y_train, epochs=3)

val_loss, val_acc = model.evaluate(x_test,y_test)
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
