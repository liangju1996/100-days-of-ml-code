# day4 Logistic Regression 逻辑回归
[逻辑回归详解](https://blog.csdn.net/liulina603/article/details/78676723)
### 逻辑回归是什么
* 被用来处理不同的分类问题(变量为分类变量(eg.患病，未患病...))
* 目的是预测当前被观察的对象属于哪个组
* 提供一个离散的二进制的输出结果
* 一个简单的例子：判断一个人是否会在即将到来的选举中进行投票
### 逻辑回归是如何工作的
* 使用基础逻辑函数通过估算概率来测量因变量（我们想要预测的标签）和一个或者多个自变量之间的关系
### 做出预测
* 这些预测值必须转换为二进制数，以便实际中进行预测，这是逻辑函数的任务，也被称为sigmoid函数。
* 然后使用阈值分类器将（0，1）范围的值转化成0和1的值来表示结果
### Sigmlid函数
* Sigmoid函数是一个S形曲线，可以实现将任意真实值映射为阈值范围为0-1的值，但不局限于0-1  
 $$\logit(p)=ln\frac{{p}/{1-p}}={a_0+a_1x_1+a_2x_2+...+a_nx_n}$$
* 得到所需的Sigmoid函数后，接下来只需要和前面的线性回归一样，拟合出该式中n个参数即可。
逻辑回归示例  
![逻辑回归示例](https://github.com/liangju1996/100-days-of-ml-code/blob/master/图片/day4.png)

### 逻辑回归vs线性回归
* 逻辑回归给出离散的输出结果,通常是处理因变量是连续变量的问题
* 线性回归给出的是连续的,因变量是定性变量(eg.患病，未患病...)

# Day7 代码实现
## Step1 导入库
[matplotlib.pyplot](http://baijiahao.baidu.com/s?id=1579894571835817104&wfr=spider&for=pc)
```python
import numpy as np
'''NumPy(Numerical Python) 是 Python 语言的一个扩展程序库，支持大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库.
Numpy是Python的一个科学计算的库，提供了矩阵运算的功能，其一般与 SciPy（Scientific Python）和 Matplotlib（绘图库）一起使用.
'''
import matplotlib.pyplot as plt
'''matplotlib.pyplot,matplotlib是python上的一个2D绘图库, matplotlib下的模块pyplot是一个有命令样式的函数集合，
matplotlib.pyplot是为我们对结果进行图像化作准备的。matplotlib.pyplot是一个命令型函数集合，它可以让我们像使用MATLAB一样使用matplotlib。
pyplot中的每一个函数都会对画布图像作出相应的改变，如创建画布、在画布中创建一个绘图区、在绘图区上画几条线、给图像添加文字说明等。'''
import pandas as pd
'''pandas 是基于NumPy 的一种工具，该工具是为了解决数据分析任务而创建的。Pandas 纳入了大量库和一些标准的数据模型，
提供了高效地操作大型数据集所需的工具。pandas提供了大量能使我们快速便捷地处理数据的函数和方法。
你很快就会发现，它是使Python成为强大而高效的数据分析环境的重要因素之一。
'''
```
### 导入数据集
```python
dataset = pd.read_csv('Social_Network_Ads.csv')
# input_data.read_data_sets()读取的是压缩包，一定不要解压
X = dataset.iloc[:, [2,3]].values
Y = dataset.iloc[:, 4].values
```
### 将数据集分成训练集和测试集
[sklearn.model_selection](https://scikit-learn.org/stable/modules/cross_validation.html)
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
'''
参数解释：
train_data：被划分的样本特征集
train_target：被划分的样本标签
test_size：如果是浮点数，在0-1之间，表示样本占比；如果是整数的话就是样本的数量
random_state：是随机数的种子。
随机数种子：其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。
比如你每次都填1，其他参数一样的情况下你得到的随机数组是一样的。但填0或不填，每次都会不一样。
随机数的产生取决于种子，随机数和种子之间的关系遵从以下两个规则：
种子不同，产生不同的随机数；种子相同，即使实例不同也产生相同的随机数。
'''
```

### 特征缩放
[数据预处理方式（去均值、归一化、PCA降维）](https://blog.csdn.net/maqunfi/article/details/82252480)
```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# StandardScaler作用：去均值(各维度都减对应维度的均值，使得输入数据各个维度都中心化为0，进行去均值的原因是因为如果不去均值的话会容易拟合)
# 和方差归一化(一种是最值归一化，比如把最大值归一化成1，最小值归一化成-1；或把最大值归一化成1，最小值归一化成0。适用于本来就分布在有限范围内的数据。        
# 另一种是均值方差归一化，一般是把均值归一化成0，方差归一化成1。适用于分布没有明显边界的情况。)。且是针对每一个特征维度来做的，而不是针对样本。
X_train = sc.fit_transform(X_train)
# fit_transform() 求得训练集X的均值啊，方差啊，最大值啊，最小值啊这些训练集的固有属性
# 然后对该trainData进行转换transform，从而实现数据的标准化、归一化等等。
X_test = sc.transform(X_test)
```

## Step2 逻辑回归模型

这项工作的库将会是一个线性模型库，之所以被称为线性是因为逻辑回归是一个线性分类器，这意味着
我们在二维空间中，我们两类用户（购买和不购买）将被一条直线分割。然后导入逻辑回归类。接下来
我们将创建该类的对象，它将作为我们训练集的分类器。
### 将逻辑回归应用于训练集
[sklearn.linear_model.LinearRegression官方手册](https://scikitlearn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)  
sklearn.linear_model模型实现了广义线性模型，包括线性回归、Ridge回归、Bayesian回归等。    
[sklearn.linear_model之LinearRegression](https://blog.csdn.net/jingyi130705008/article/details/78163955)    
[sklearn.linear_model.LogisticRegression官方手册](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)  
[(sklearn)逻辑回归linear_model.LogisticRegression用法](https://blog.csdn.net/mrxjh/article/details/78499801)  
```python
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
# 分类器
```

# Step3 预测
# 预测测试结果
```python
y_pred = classifier.predict(X_test)
```

## Step4 评估预测
我们预测了测试集，现在我么将评估逻辑回归模型是否正确的学习和理解。因此这个混淆矩阵将包含我们模型的正确和错误的预测。
### 生成混淆矩阵
[sklearn中的模型评估-构建评估函数 ](https://www.cnblogs.com/harvey888/p/6964741.html)  
sklearn.metric提供了一些函数，用来计算真实值与预测值之间的预测误差：  
以_score结尾的函数，返回一个最大值，越高越好  
以_error结尾的函数，返回一个最小值，越小越好；  
如果使用make_scorer来创建scorer时，将greater_is_better设为False
```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# confusion_matrix函数通过计算混淆矩阵，用来计算分类准确率。
```

### 可视化
[matplotlib.colors(ListedColormap)](https://blog.csdn.net/zhaogeng111/article/details/78419015)  
[numpy.meshgrid():生成网格点坐标矩阵](https://blog.csdn.net/lllxxq141592654/article/details/81532855)

```python
from matplotlib.colors import ListedColormap
X_set, y_set = X_train,y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:,0].min()-1, stop = X_set[:, 0].max()+1, step = 0.01),
                    np.arange(start = X_set[:,1].min()-1, stop = X_set[:, 1].max()+1, setp = 0.01))

'''返回值： np.arange()函数返回一个有终点和起点的固定步长的排列，如[1,2,3,4,5]，起点是1，终点是5，步长为1。 
参数个数情况： np.arange()函数分为一个参数，两个参数，三个参数三种情况 
 1）一个参数时，参数值为终点，起点取默认值0，步长取默认值1。 
 2）两个参数时，第一个参数为起点，第二个参数为终点，步长取默认值1。 
 3）三个参数时，第一个参数为起点，第二个参数为终点，第三个参数为步长。其中步长支持小数。
'''
plt.contourf(X1, X2, classifier.predict(np.array([X1, ravel(), X2.ravel()]).T).reshape(X1. shape),
             alpha = 0.75,cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0],X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i),label = j)
plt.titel('LOGISTIC(Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

X_set, y_set = X_test,y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min()-1, stop = X_set[:, 0].max()+1, step = 0.01),
                     np.arange(start = X_set[:, 1].min()-1, stop = X_set[:, 1].max()+1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1, ravel(),X2,ravel()]).T).reshape(X1.shape),
             alpha = 0.75,cmp = ListedColormap(('red','green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('LOGISTIC(Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
```
