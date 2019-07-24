# CNN 相关知识 
1.Convolution(卷积) -> Pooling(池化) -> Convolution -> Pooling -> Fully Connected Layer(全连接层) -> Output  

2.Convolution（卷积）是获取原始数据并从中创建特征映射的行为。

3.Pooling(池化)是下采样，通常以“max-pooling”的形式，我们选择一个区域，然后在该区域中取最大值，这将成为整个区域的新值。

4.Fully Connected Layers(全连接层)是典型的神经网络，其中所有节点都“完全连接”。卷积层不像传统的神经网络那样完全连接。  

卷积：我们将采用某个窗口，并在该窗口中查找要素,该窗口的功能现在只是新功能图中的一个像素大小的功能，但实际上我们将有多层功能图。
接下来，我们将该窗口滑过并继续该过程,继续此过程，直到覆盖整个图像。

池化：最常见的池化形式是“最大池化”，其中我们简单地获取窗口中的最大值，并且该值成为该区域的新值。

全连接层：每个卷积和池化步骤都是隐藏层。在此之后，我们有一个完全连接的层，然后是输出层。
完全连接的层是典型的神经网络（多层感知器）类型的层，与输出层相同。


[keras中几个重要函数用法](https://blog.csdn.net/u012969412/article/details/70882296)
## CNN 实现代码
```python
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
# cifar10是一个包含60000张图片的数据集。其中每张照片为32*32的彩色照片，每个像素点包括RGB三个数值，数值范围 0 ~ 255。所有照片分属10个不同的类别，
# 分别是 'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck',
# 其中五万张图片被划分为训练集，剩下的一万张图片属于测试集。
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# ImageDataGenerator是keras.preprocessing.image模块中的图片生成器，同时也可以在batch中对数据进行增强，扩充数据集大小，增强模型的泛化能力。
比如进行旋转，变形，归一化等等。
from tensorflow.keras.models import Sequential
# Sequential序列惯性模型，序贯模型是函数式模型的简略版，为最简单的线性、从头到尾的结构顺序，不分叉。
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
# Dense是全连接层，
# Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。
# Activation 是激活函数。
# dropout顾名思义就是丢弃，丢弃的是每一层的某些神经元。在DNN深度网络中过拟合问题一直存在，dropout技术可以在一定程度上防止网络的过拟合。

from tensorflow.keras.layers import Conv2D, MaxPooling2D
# Conv2D （卷积层）,MaxPooling2D（池化层）

import pickle

pickle_in = open("E:/pycharm/practice/kagglecatsanddogs_3367a/PetImages/X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("E:/pycharm/practice/kagglecatsanddogs_3367a/PetImages/y.pickle","rb")
y = pickle.load(pickle_in)

X = X/255.0

model = Sequential()

model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# pool_size：整数或长为2的整数tuple，代表在两个方向（竖直，水平）上的下采样因子，如取（2，2）将使图片在两个维度上均变为原长的一半。
# 为整数意为各个维度值相同且为该数字。

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, y, batch_size=32, epochs=3, validation_split=0.3)
# 见链接
```
[model.fit](https://blog.csdn.net/a1111h/article/details/82148497)
