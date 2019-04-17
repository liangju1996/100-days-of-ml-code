# day3 Multiple Linear Regression 多元线性回归
当Y值的影响因素不唯一时，使用多元线性回归；反之，使用一元
[学习线性回归与逻辑回归]（https://blog.csdn.net/m0_37622530/article/details/80949562）
## Step1: Data Preprocessing 数据预处理
### Importing the libraries 导入库
```python
import pandas as pd
import numpy as np
```
### Importing the dataset 导入数据集
```python
dataset = pd.read_csv('../GitHub clone/100-Days-Of-ML-Code/datasets/50_Startups.csv')
print(dataset)
X = dataset.iloc[ : ,  :-1].values
Y = dataset.iloc[ : ,  4].values
print('X')
print(X)
print('Y')
print(Y)
```
### Encoding Catagorical data 将类别数据数字化
[scikit-learn官网](https://scikit-learn.org/stable/index.html)
[独热编码](https://blog.csdn.net/a595130080/article/details/64442800)
```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[ : , 3] = labelencoder.fit_transform(X[ : , 3])
# 四列数据都输入OneHotEncoder，但是做变换的只有第四列，转换得到的那三列categorical数据（OneHotEncode前是最后一列）被放在头三列，而不是最后三列。
onehotencoder = OneHotEncoder(categorical_features = [3])  # 指定第四列（index为3）
X = onehotencoder.fit_transform(X).toarray()

# toarray()的作用：将矩阵变成二维数组？？？有一个方法实在不会时，可在代码中分别加上和取消这个方法看两者输出有什么差别
```
### Avoiding Dummy Variable Trap 躲避虚拟变量陷阱
虚拟变量其实算不上一种变量类型（比如连续变量、分类变量等），确切地说，是一种将多分类变量转换为二分变量的一种形式。
比如预测体重w，输入的变量由身高h和性别、人种。其中性别（男or女，暂不考虑复杂情况0）、人种（黄or白or黑，暂不考虑混血等复杂情况）不是连续变量。此时的回归模型为：

`w = a + b*h +c*is_man +d*is_yellow + e*is_white`

该模型为加法模型。其中is_man，is_yellow，is_white都只能取0或1（能取n个值的变量，在式子中由n-1项表示；比如人种一共三类，则式子中用is_yellow，is_white两个变量表示；否则会造成多重共线性）。

```python
X = X[ : , 1: ]
```

### Slitting the dataset into the Training set and Test set 拆分数据集为训练集和测试集
```python
from sklearn.model_selection import train_test_split
# 原来是sklearn.cross_validation， 现改为sklearn.model_selection，，，原因见day2
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
```

## Step 2: Fitting Multiple Linear Regression to the Training set 在训练集上训练多元线性回归模型
```python
from sklearn.linear_model import  LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
```

## Step3: Predicting the Test set results 在测试集上预测结果
```python
y_pred = regressor.predict(X_test)
```
