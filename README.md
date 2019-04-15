# Day 1: Data Prepocessing(数据预处理)

## Step 1: Importing the libraries(导入库)
```python 
  import numpy as np
  '''
  NumPy(Numerical Python) 是 Python 语言的一个扩展程序库，支持大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库。Numpy是Python的一个科学计算的库，提供了矩阵运算的功能，其一般与 SciPy（Scientific Python）和 Matplotlib（绘图库）一起使用.
  '''
  import pandas as pd
  ''' pandas 是基于NumPy 的一种工具，该工具是为了解决数据分析任务而创建的。Pandas 纳入了大量库和一些标准的数据模型，提供了高效地操作大型数据集所需的工具。pandas提供了大量能使我们快速便捷地处理数据的函数和方法。你很快就会发现，它是使Python成为强大而高效的数据分析环境的重要因素之一。
  '''
```
## Step 2: Importing dataset(导入数据集)
```python
  import sklearn
  """Scikit-learn(sklearn)是机器学习中常用的第三方模块，对常用的机器学习方法进行了封装，包括回归(Regression)、降维(Dimensionality Reduction)、分类(Classfication)、聚类(Clustering)等方法。
  """
  dataset = pd.read_csv('../GitHub clone/100-Days-Of-ML-Code/datasets/Data.csv')
  # 将自变量（3列）和因变量（1列）拆开，拆成一个矩阵和一个向量。
  # 取除了最后一列的所有数据
  X = dataset.iloc[ : , :-1].values
  # 数组
  # 索引右边第一个从零开始，左边第一个从-1开始，每次从右边开始读，读到左边显示数字的前一个

  # loc函数：通过行（列）索引 "Index" 中的具体值来取行数据（如取"Index"为"A"的行）根据DataFrame的具体标签选取列（行）
  # iloc函数：通过行号来取行（列）数据（如取第二行数据）根据标签的所在位置，从0开始计数，选取列（行）

  # 取第三行的所有数据
  Y = dataset.iloc[ : , 3].values

  print("Step 2: Importing dataset")
  print("X")
  print(X)
  print("Y")
  print(Y)
```
## Step 3: Handling the missing data(处理缺失数据)
```python
from sklearn.preprocessing import Imputer
imputer = sklearn.preprocessing.Imputer(missing_values ="NaN", strategy ="mean", axis = 0)
"""
填补缺失值：sklearn.preprocessing.Imputer(missing_values=’NaN’, strategy=’mean’, axis=0, verbose=0, copy=True)
主要参数说明：
missing_values：缺失值，可以为整数或NaN(缺失值numpy.nan用字符串‘NaN’表示)，默认为NaN(not a number)
strategy：替换策略，字符串，默认用均值‘mean’替换
  ①若为mean时，用特征列的均值替换
  ②若为median时，用特征列的中位数替换
  ③若为most_frequent时，用特征列的众数替换
axis：指定轴数，默认axis=0代表列，axis=1代表行
  axis为0，如果列中全是缺失值，无法fit的情况就被丢弃， 
  axis为1，如果行中全是缺失值，就发生错误。
copy：默认True，可不写，设置为True代表不在原数据集上修改，设置为False时，就地修改，存在如下情况时，即使设置为False时，也不会就地修改
  ①X不是浮点值数组
  ②X是稀疏且missing_values=0
  ③axis=0且X为CRS矩阵
  ④axis=1且X为CSC矩阵
verbose：整数，可不写，默认为零，控制精度
"""
imputer = imputer.fit(X[ : , 1:3])
# fit() 求得训练集X的均值啊，方差啊，最大值啊，最小值啊这些训练集的固有属性
X[ : , 1:3] = imputer.transform(X[ : , 1:3])
# imputer.transform()在Fit的基础上，进行标准化，降维，归一化等操作
print("---------------------")
print("Step 3: Handling the missing data")
print("step2")
print("X")
print(X)
```
## Step 4: Encoding categorical data(解析分类数据)
```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
# LabelEncoder可以将标签分配一个0—n_classes-1之间的编码 将各种标签分配一个可数的连续编号：
# fit_transform(trainData)  对部分数据先拟合fit，找到该part的整体指标，如均值、方差、最大值最小值等等（根据具体转换的目的），然后对该trainData进行转换transform，从而实现数据的标准化、归一化等等。
X[ : , 0] = labelencoder_X.fit_transform(X[ : , 0])
# Creating a dummy variable(创建虚拟变量)
onehotencoder = OneHotEncoder(categorical_features = [0])
# 独热码，在英文文献中称做 one-hot code, 直观来说就是有多少个状态就有多少比特，而且只有一个比特为1，其他全为0的一种码制。
# categorical_features是需要独热编码的列索引，n_values是对应categorical_features中各列下类别的数目，也就是原来的列拓展出新的列数。
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y =  labelencoder_Y.fit_transform(Y)
print("---------------------")
print("Step 4: Encoding categorical data")
print("X")
print(X)
print("Y")
print(Y)
```
## Step 5: Splitting the datasets into training sets and Test sets(拆分数据集为训练集合和测试集合)
```python
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X , Y , test_size = 0.2, random_state = 0)
""" train_test_split函数用于将矩阵随机划分为训练子集和测试子集，并返回划分好的训练集测试集样本和训练集测试集标签。
格式：
X_train,X_test, y_train, y_test =cross_validation.train_test_split(train_data,train_target,test_size=0.3, random_state=0)
 
参数解释：
train_data：被划分的样本特征集
train_target：被划分的样本标签
test_size：如果是浮点数，在0-1之间，表示样本占比；如果是整数的话就是样本的数量
random_state：是随机数的种子。
随机数种子：其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。比如你每次都填1，其他参数一样的情况下你得到的随机数组是一样的。但填0或不填，每次都会不一样。
随机数的产生取决于种子，随机数和种子之间的关系遵从以下两个规则：
种子不同，产生不同的随机数；种子相同，即使实例不同也产生相同的随机数。
"""
print("---------------------")
print("Step 5: Splitting the datasets into training sets and Test sets")
print("X_train")
print(X_train)
print("X_test")
print(X_test)
print("Y_train")
print(Y_train)
print("Y_test")
print(Y_test)
```
## Step 6: Feature Scaling(特征量化)
```python
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
# 数据在前处理的时候，经常会涉及到数据标准化。将现有的数据通过某种关系，映射到某一空间内。常用的标准化方式是,减去平均值，然后通过标准差映射到均至为0的空间内。系统会记录每个输入参数的平均数和标准差，以便数据可以还原。
# 很多ML的算法要求训练的输入参数的平均值是0并且有相同阶数的方差例如:RBF核的SVM，L1和L2正则的线性回归sklearn.preprocessing.StandardScaler能够轻松的实现上述功能。
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
print("---------------------")
print("Step 6: Feature Scaling")
print("X_train")
print(X_train)
print("X_test")
print(X_test)
```
