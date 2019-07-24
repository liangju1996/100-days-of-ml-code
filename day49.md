##　合并数据集：Concat与Append操作
将不同的数据源进行合并是数据科学中最有趣的事情之一，这既包括将两个不同的数据集
非常简单地拼接在一起，也包括用数据库那样的连接（join）与合并（merge）操作处理有
重叠字段的数据集。Series 与 DataFrame 都具备这类操作，Pandas 的函数与方法让数据合
并变得快速简单。
先来用 pd.concat 函数演示一个 Series 与 DataFrame 的简单合并操作。之后，我们将介绍
Pandas 中更复杂的 merge 和 join 内存数据合并操作。
首先导入 Pandas 和 NumPy：
```python
In[1]: import pandas as pd
 import numpy as np
 ```
简单起见，定义一个能够创建 DataFrame 某种形式的函数，后面将会用到：
```python
In[2]: def make_df(cols, ind):
 """一个简单的DataFrame"""
 data = {c: [str(c) + str(i) for i in ind]
 for c in cols}
 return pd.DataFrame(data, ind)
 # DataFrame示例
 make_df('ABC', range(3))
Out[2]: A B C
 0 A0 B0 C0
 1 A1 B1 C1
 2 A2 B2 C2
```
### 　知识回顾：NumPy数组的合并
合 并 Series 与 DataFrame 与合并 NumPy 数 组 基 本 相 同， 后 者 通 过 2.2 节中介绍的
np.concatenate 函数即可完成。你可以用这个函数将两个或两个以上的数组合并成一个数组。
```python
In[4]: x = [1, 2, 3]
 y = [4, 5, 6]
 z = [7, 8, 9]
 np.concatenate([x, y, z])
Out[4]: array([1, 2, 3, 4, 5, 6, 7, 8, 9])
```
第一个参数是需要合并的数组列表或元组。还有一个 axis 参数可以设置合并的坐标轴
方向：
```python
In[5]: x = [[1, 2],
 [3, 4]]
 np.concatenate([x, x], axis=1)
Out[5]: array([[1, 2, 1, 2],
 [3, 4, 3, 4]])
 
### 　通过pd.concat实现简易合并
Pandas 有一个 pd.concat() 函数与 np.concatenate 语法类似，但是配置参数更多，功能也
更强大：
```python
# Pandas 0.18版中的函数签名
pd.concat(objs, axis=0, join='outer', join_axes=None, ignore_index=False,
 keys=None, levels=None, names=None, verify_integrity=False,
 copy=True)
 ```
pd.concat() 可以简单地合并一维的 Series 或 DataFrame 对象，与 np.concatenate() 合并
数组一样：
```python
In[6]: ser1 = pd.Series(['A', 'B', 'C'], index=[1, 2, 3])
 ser2 = pd.Series(['D', 'E', 'F'], index=[4, 5, 6])
 pd.concat([ser1, ser2])
Out[6]: 1 A
 2 B
 3 C
 4 D
 5 E
 6 F
 dtype: object
 ```
它也可以用来合并高维数据，例如下面的 DataFrame：
```python
In[7]: df1 = make_df('AB', [1, 2])
 df2 = make_df('AB', [3, 4])
 print(df1); print(df2); print(pd.concat([df1, df2])) 
df1 df2 pd.concat([df1, df2])
 A B A B A B
 1 A1 B1 3 A3 B3 1 A1 B1
 2 A2 B2 4 A4 B4 2 A2 B2
 3 A3 B3
 4 A4 B4
 ```
默认情况下，DataFrame 的合并都是逐行进行的（默认设置是 axis=0）。与 np.concatenate()
一样，pd.concat 也可以设置合并坐标轴，例如下面的示例：
```python
In[8]: df3 = make_df('AB', [0, 1])
 df4 = make_df('CD', [0, 1])
 print(df3); print(df4); print(pd.concat([df3, df4], axis='col'))
df3 df4 pd.concat([df3, df4], axis='col')
 A B C D A B C D
 0 A0 B0 0 C0 D0 0 A0 B0 C0 D0
 1 A1 B1 1 C1 D1 1 A1 B1 C1 D1
 ```
这里也可以使用 axis=1，效果是一样的。但是用 axis='col' 会更直观。
#### 1. 索引重复
np.concatenate 与 pd.concat 最主要的差异之一就是 Pandas 在合并时会保留索引，即使索
引是重复的！例如下面的简单示例：
```python
In[9]: x = make_df('AB', [0, 1])
 y = make_df('AB', [2, 3])
 y.index = x.index # 复制索引
 print(x); print(y); print(pd.concat([x, y]))
x y pd.concat([x, y])
 A B A B A B
 0 A0 B0 0 A2 B2 0 A0 B0
 1 A1 B1 1 A3 B3 1 A1 B1
 0 A2 B2
 1 A3 B3
 ```
你会发现结果中的索引是重复的。虽然 DataFrame 允许这么做，但结果并不是我们想要的。
pd.concat() 提供了一些解决这个问题的方法。
(1) 捕捉索引重复的错误。如果你想要检测 pd.concat() 合并的结果中是否出现了重复的索
引，可以设置 verify_integrity 参数。将参数设置为 True，合并时若有索引重复就会
触发异常。下面的示例可以让我们清晰地捕捉并打印错误信息：
```python
In[10]: try:
 pd.concat([x, y], verify_integrity=True)
 except ValueError as e:
 print("ValueError:", e)
ValueError: Indexes have overlapping values: [0, 1]
(2) 忽略索引。有时索引无关紧要，那么合并时就可以忽略它们，可以通过设置 ignore_
index 参数来实现。如果将参数设置为 True，那么合并时将会创建一个新的整数索引。
```python
In[11]: print(x); print(y); print(pd.concat([x, y], ignore_index=True))
x y pd.concat([x, y], ignore_index=True)
 A B A B A B
 0 A0 B0 0 A2 B2 0 A0 B0
 1 A1 B1 1 A3 B3 1 A1 B1
 2 A2 B2
 3 A3 B3
 ```
(3) 增加多级索引。另一种处理索引重复的方法是通过 keys 参数为数据源设置多级索引标
签，这样结果数据就会带上多级索引：
```python
In[12]: print(x); print(y); print(pd.concat([x, y], keys=['x', 'y']))
x y pd.concat([x, y], keys=['x', 'y'])
 A B A B A B
 0 A0 B0 0 A2 B2 x 0 A0 B0
 1 A1 B1 1 A3 B3 1 A1 B1
 y 0 A2 B2
 1 A3 B3
 ```
示例合并后的结果是多级索引的 DataFrame，可以用 3.6 节介绍的方法将它转换成我们需要
的形式。
#### 2. 类似join的合并
前面介绍的简单示例都有一个共同特点，那就是合并的 DataFrame 都是同样的列名。而在
实际工作中，需要合并的数据往往带有不同的列名，而 pd.concat 提供了一些选项来解决
这类合并问题。看下面两个 DataFrame，它们的列名部分相同，却又不完全相同：
In[13]: df5 = make_df('ABC', [1, 2])
 df6 = make_df('BCD', [3, 4])
 print(df5); print(df6); print(pd.concat([df5, df6])
df5 df6 pd.concat([df5, df6])
 A B C B C D A B C D
 1 A1 B1 C1 3 B3 C3 D3 1 A1 B1 C1 NaN
 2 A2 B2 C2 4 B4 C4 D4 2 A2 B2 C2 NaN
 3 NaN B3 C3 D3
 4 NaN B4 C4 D4
 ```
 默认情况下，某个位置上缺失的数据会用 NaN 表示。如果不想这样，可以用 join 和 join_
axes 参数设置合并方式。默认的合并方式是对所有输入列进行并集合并（join='outer'），
当然也可以用 join='inner' 实现对输入列的交集合并：
```python
In[14]: print(df5); print(df6);
 print(pd.concat([df5, df6], join='inner'))
df5 df6 pd.concat([df5, df6], join='inner')
 A B C B C D B C
 1 A1 B1 C1 3 B3 C3 D3 1 B1 C1
 2 A2 B2 C2 4 B4 C4 D4 2 B2 C2
 3 B3 C3
 4 B4 C4
```
另一种合并方式是直接确定结果使用的列名，设置 join_axes 参数，里面是索引对象构成
的列表（是列表的列表）。如下面示例所示，将结果的列名设置为第一个输入的列名：
```python
In[15]: print(df5); print(df6);
 print(pd.concat([df5, df6], join_axes=[df5.columns]))
df5 df6 pd.concat([df5, df6], join_axes=[df5.columns])
 A B C B C D A B C
 1 A1 B1 C1 3 B3 C3 D3 1 A1 B1 C1
 2 A2 B2 C2 4 B4 C4 D4 2 A2 B2 C2
 3 NaN B3 C3
 4 NaN B4 C4
 ```
pd.concat 的合并功能可以满足你在合并两个数据集时的许多需求，操作时请记住这一点。
#### 3. append()方法
因为直接进行数组合并的需求非常普遍，所以 Series 和 DataFrame 对象都支持 append 方
法，让你通过最少的代码实现合并功能。例如，你可以使用 df1.append(df2)，效果与
pd.concat([df1, df2]) 一样：
```python
In[16]: print(df1); print(df2); print(df1.append(df2))
df1 df2 df1.append(df2)
 A B A B A B
 1 A1 B1 3 A3 B3 1 A1 B1
 2 A2 B2 4 A4 B4 2 A2 B2
 3 A3 B3
 4 A4 B4
 ```
需要注意的是，与 Python 列表中的 append() 和 extend() 方法不同，Pandas 的 append() 不
直接更新原有对象的值，而是为合并后的数据创建一个新对象。因此，它不能被称之为一
个非常高效的解决方案，因为每次合并都需要重新创建索引和数据缓存。总之，如果你需
要进行多个 append 操作，还是建议先创建一个 DataFrame 列表，然后用 concat() 函数一次
性解决所有合并任务。
