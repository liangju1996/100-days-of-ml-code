```python
import numpy as np
import matplotlib.pyplot as plt
import os
# 操作系统接口模块：os模块提供了多数操作系统的功能接口函数。
# 当os模块被导入后，它会自适应于不同的操作系统平台，根据不同的平台进行相应的操作，
# 在python编程时，经常和文件、目录打交道，所以离不了os模块。
# https://docs.python.org/zh-cn/3.7/library/os.html
import cv2
# cv2 python的图像处理模块
from tqdm import tqdm
# tqdm是python中很常用的模块，它的作用就是在终端上出现一个进度条，使得代码进度可视化。

#
DATADIR = "E:/pycharm/practice/kagglecatsanddogs_3367a/PetImages" # 数据集的路径，请根据需要修改

CATEGORIES = ["Dog", "Cat"]

# for循环用于遍历CATEGORIES，把取出来的图片存储在变量category
for category in CATEGORIES:
    path = os.path.join(DATADIR,category)  # 创建路径
    # 迭代遍历每个图片，在python中可以使用os.listdir()函数获得指定目录中的内容。
    for img in os.listdir(path):
        # 用来读取图片文件中的数据
        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)
        # 转化成array
        plt.imshow(img_array, cmap='gray')  # 转换成图像展示
        plt.show()  # display!

        break  # 我们作为演示只展示一张，所以直接break了
    break  #同上
    print(img_array)
    print(img_array.shape)

IMG_SIZE = 50
#resize图像缩放函数
new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap='gray')
plt.show()

# size设置成50有些模糊，尝试下100
IMG_SIZE = 100
new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap='gray')
plt.show()

training_data = []
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR,category)
        # 得到分类，其中 0=dog 1=cat
        class_num = CATEGORIES.index(category)
        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)
                # 大小转换
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                # 加入训练数据中
                training_data.append([new_array, class_num])
                # 为了保证输出是整洁的
            except Exception as e:
                pass
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))

create_training_data()
# len() 方法返回对象（字符、列表、元组等）长度或项目个数。
#print(len(training_data))
# shuffle()函数是将training_data里的数据随机打乱
print(len(training_data))

import random

random.shuffle(training_data)
# 遍历training_data里的第0列到第9列
for sample in training_data[:10]:
    print(sample[1])

X = []
y = []

for features,label in training_data:
    # 在X里面追加上features
    X.append(features)
    y.append(label)
# reshape函数用处：将一个矩阵重新生成任意维度的矩阵（元素个数内）
# 行数
print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

import pickle
# 加工数据的，可以用来存取结构化数据，也就是说pickle可以把字典、列表等结构化数据存到本地文件，
# 读取后返回的还是字典、列表等结构化数据。
# 以二进制的方式写入打开的文件中
pickle_out = open("E:/pycharm/practice/kagglecatsanddogs_3367a/PetImages/X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("E:/pycharm/practice/kagglecatsanddogs_3367a/PetImages/y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()
# We can always load it in to our current script, or a totally new one by doing:
# 我们总是可以将它加载到当前脚本中，或者通过doing加载一个全新的脚本
pickle_in = open("E:/pycharm/practice/kagglecatsanddogs_3367a/PetImages/X.pickle","rb")
# pickle.load()方法，是反序列化对象，将pickle_in中的数据解析为一个python对象
X = pickle.load(pickle_in)

pickle_in = open("E:/pycharm/practice/kagglecatsanddogs_3367a/PetImages/y.pickle","rb")
y = pickle.load(pickle_in)

```








