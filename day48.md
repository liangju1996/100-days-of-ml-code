# Pandas数据处理
Pandas 是在 NumPy 基础上建立的新程序库，提供了一种高效的 DataFrame
数据结构。DataFrame 本质上是一种带行标签和列标签、支持相同类型数据和缺失值的多
维数组。Pandas 不仅为带各种标签的数据提供了便利的存储界面，还实现了许多强大的操
作。建立在 NumPy 数组结构上的
Pandas，尤其是它的 Series 和 DataFrame 对象，为数据科学家们处理那些消耗大量时间的
“数据清理”（data munging）任务提供了捷径。
如果从底层视角观察 Pandas 对象，可以把它们看成增强版的 NumPy 结构化数组，行列都
不再只是简单的整数索引，还可以带上标签。
虽然
Pandas 在基本数据结构上实现了许多便利的工具、方法和功能，但是后面将要介绍的每
一个工具、方法和功能几乎都需要我们理解基本数据结构的内部细节。因此，在深入学习
Pandas 之前，先来看看 Pandas 的三个基本数据结构：Series、DataFrame 和 Index。
## pandas对象简介
### Pandas的Series对象
Pandas 的 Series 对象是一个带索引数据构成的一维数组。可以用一个数组创建 Series 对
象，如下所示：
```python
In[2]: data = pd.Series([0.25, 0.5, 0.75, 1.0])
 data 

Out[2]: 0 0.25
 1 0.50
 2 0.75
 3 1.00
 dtype: float64
  ```
从上面的结果中，你会发现 Series 对象将一组数据和一组索引绑定在一起，我们可以通过
values 属性和 index 属性获取数据。values 属性返回的结果与 NumPy 数组类似：
```python
In[3]: data.values
Out[3]: array([ 0.25, 0.5 , 0.75, 1. ])
 ```
index 属性返回的结果是一个类型为 pd.Index 的类数组对象，我们将在后面的内容里详细
介绍它：
```python
In[4]: data.index
Out[4]: RangeIndex(start=0, stop=4, step=1)
 ```
和 NumPy 数组一样，数据可以通过 Python 的中括号索引标签获取
```python
In[5]: data[1]
Out[5]: 0.5
In[6]: data[1:3]
Out[6]: 1 0.50
 2 0.75
 dtype: float64
  ```
但是我们将会看到，Pandas 的 Series 对象比它模仿的一维 NumPy 数组更加通用、灵活。
#### 1. Serise是通用的NumPy数组
到目前为止，我们可能觉得 Series 对象和一维 NumPy 数组基本可以等价交换，但两者
间的本质差异其实是索引：NumPy 数组通过隐式定义的整数索引获取数值，而 Pandas 的
Series 对象用一种显式定义的索引与数值关联。
显式索引的定义让 Series 对象拥有了更强的能力。例如，索引不再仅仅是整数，还可以是
任意想要的类型。如果需要，完全可以用字符串定义索引：
```python
In[7]: data = pd.Series([0.25, 0.5, 0.75, 1.0],
 index=['a', 'b', 'c', 'd'])
 data
Out[7]: a 0.25
 b 0.50
 c 0.75
 d 1.00
 dtype: float64
  ```
获取数值的方式与之前一样
```python
In[8]: data['b']
Out[8]: 0.5
 ```
也可以使用不连续或不按顺序的索引：
```python
In[9]: data = pd.Series([0.25, 0.5, 0.75, 1.0],
 index=[2, 5, 3, 7])
 data
Out[9]: 2 0.25
 5 0.50
 3 0.75
 7 1.00
 dtype: float64
In[10]: data[5]
Out[10]: 0.5
 ```
#### 2. Series是特殊的字典
你可以把 Pandas 的 Series 对象看成一种特殊的 Python 字典。字典是一种将任意键映射到
一组任意值的数据结构，而 Series 对象其实是一种将类型键映射到一组类型值的数据结
构。类型至关重要：就像 NumPy 数组背后特定类型的经过编译的代码使得它在某些操作
上比普通的 Python 列表更加高效一样，Pandas Series 的类型信息使得它在某些操作上比
Python 的字典更高效。
我们可以直接用 Python 的字典创建一个 Series 对象，让 Series 对象与字典的类比更
加清晰：
```python
In[11]: population_dict = {'California': 38332521,
 'Texas': 26448193,
 'New York': 19651127,
 'Florida': 19552860,
 'Illinois': 12882135}
 population = pd.Series(population_dict)
 population
Out[11]: California 38332521
 Florida 19552860
 Illinois 12882135
 New York 19651127
 Texas 26448193
 dtype: int64
  ```
用字典创建 Series 对象时，其索引默认按照顺序排列。典型的字典数值获取方式仍然
有效：
```python
In[12]: population['California']
Out[12]: 38332521
 ```
和字典不同，Series 对象还支持数组形式的操作，比如切片：
```python
In[13]: population['California':'Illinois']
Out[13]: California 38332521
 Florida 19552860
 Illinois 12882135
 dtype: int64
 ```
我们将在 3.3 节中介绍 Pandas 取值与切片的一些技巧。
#### 3. 创建Series对象
我们已经见过几种创建 Pandas 的 Series 对象的方法，都是像这样的形式：
```python
>>> pd.Series(data, index=index)
 ```
其中，index 是一个可选参数，data 参数支持多种数据类型。
例如，data 可以是列表或 NumPy 数组，这时 index 默认值为整数序列：
```python
In[14]: pd.Series([2, 4, 6])
Out[14]: 0 2
 1 4
 2 6
 dtype: int64
  ```
data 也可以是一个标量，创建 Series 对象时会重复填充到每个索引上：
```python
In[15]: pd.Series(5, index=[100, 200, 300])
Out[15]: 100 5
 200 5
 300 5
 dtype: int64
  ```
data 还可以是一个字典，index 默认是排序的字典键：
```python
In[16]: pd.Series({2:'a', 1:'b', 3:'c'})
Out[16]: 1 b
 2 a
 3 c
 dtype: object
  ```
每一种形式都可以通过显式指定索引筛选需要的结果：
```python
In[17]: pd.Series({2:'a', 1:'b', 3:'c'}, index=[3, 2])
Out[17]: 3 c
 2 a
 dtype: object
 ```
这里需要注意的是，Series 对象只会保留显式定义的键值对。
### Pandas的DataFrame对象
Pandas 的另一个基础数据结构是 DataFrame。和上一节介绍的 Series 对象一样，DataFrame
既可以作为一个通用型 NumPy 数组，也可以看作特殊的 Python 字典。下面来分别看看。
#### 1. DataFrame是通用的NumPy数组
如果将 Series 类比为带灵活索引的一维数组，那么 DataFrame 就可以看作是一种既有灵活
的行索引，又有灵活列名的二维数组。就像你可以把二维数组看成是有序排列的一维数组
一样，你也可以把 DataFrame 看成是有序排列的若干 Series 对象。这里的“排列”指的是
它们拥有共同的索引。
下面用上一节中美国五个州面积的数据创建一个新的 Series 来进行演示：
```python
In[18]:
area_dict = {'California': 423967, 'Texas': 695662, 'New York': 141297,
 'Florida': 170312, 'Illinois': 149995}
area = pd.Series(area_dict)
area
Out[18]: California 423967
 Florida 170312
 Illinois 149995
 New York 141297
 Texas 695662
 dtype: int64
 ```
再结合之前创建的 population 的 Series 对象，用一个字典创建一个包含这些信息的二维
对象：
```python
In[19]: states = pd.DataFrame({'population': population,
 'area': area})
 states
Out[19]: area population
 California 423967 38332521
 Florida 170312 19552860
 Illinois 149995 12882135
 New York 141297 19651127
 Texas 695662 26448193
 ```
和 Series 对象一样，DataFrame 也有一个 index 属性可以获取索引标签：
```python
In[20]: states.index
Out[20]:
Index(['California', 'Florida', 'Illinois', 'New York', 'Texas'], dtype='object')
```
另外，DataFrame 还有一个 columns 属性，是存放列标签的 Index 对象：
```python
In[21]: states.columns
Out[21]: Index(['area', 'population'], dtype='object')
```
因此 DataFrame 可以看作一种通用的 NumPy 二维数组，它的行与列都可以通过索引获取。
#### 2. DataFrame是特殊的字典
与 Series 类似，我们也可以把 DataFrame 看成一种特殊的字典。字典是一个键映射一个
值，而 DataFrame 是一列映射一个 Series 的数据。例如，通过 'area' 的列属性可以返回
包含面积数据的 Series 对象：
```python
In[22]: states['area']
Out[22]: California 423967
 Florida 170312
 Illinois 149995
 New York 141297
 Texas 695662
 Name: area, dtype: int64
 ```
这里需要注意的是，在 NumPy 的二维数组里，data[0] 返回第一行；而在 DataFrame 中，
data['col0'] 返回第一列。因此，最好把 DataFrame 看成一种通用字典，而不是通用数
组，即使这两种看法在不同情况下都是有用的。
#### 3. 创建DataFrame对象
Pandas 的 DataFrame 对象可以通过许多方式创建，这里举几个常用的例子。
(1) 通过单个 Series 对象创建。DataFrame 是一组 Series 对象的集合，可以用单个 Series
创建一个单列的 DataFrame：
```python
In[23]: pd.DataFrame(population, columns=['population'])
Out[23]: population
 California 38332521
 Florida 19552860
 Illinois 12882135
 New York 19651127
 Texas 26448193
 ```
(2) 通过字典列表创建。任何元素是字典的列表都可以变成 DataFrame。用一个简单的列表
综合来创建一些数据：
```python
In[24]: data = [{'a': i, 'b': 2 * i}
 for i in range(3)]
 pd.DataFrame(data)
Out[24]: a b
 0 0 0
 1 1 2
 2 2 4
 ```
即使字典中有些键不存在，Pandas 也会用缺失值 NaN（不是数字，not a number）来表示：
```python
In[25]: pd.DataFrame([{'a': 1, 'b': 2}, {'b': 3, 'c': 4}])
Out[25]: a b c
 0 1.0 2 NaN
 1 NaN 3 4.0
 ```
(3) 通过 Series 对象字典创建。就像之前见过的那样，DataFrame 也可以用一个由 Series
对象构成的字典创建：
```python
In[26]: pd.DataFrame({'population': population,
 'area': area})
Out[26]: area population
 California 423967 38332521
 Florida 170312 19552860
 Illinois 149995 12882135
 New York 141297 19651127
 Texas 695662 26448193
 ```
(4) 通过 NumPy 二维数组创建。假如有一个二维数组，就可以创建一个可以指定行列索引
值的 DataFrame。如果不指定行列索引值，那么行列默认都是整数索引值：
```python
In[27]: pd.DataFrame(np.random.rand(3, 2),
 columns=['foo', 'bar'],
 index=['a', 'b', 'c'])
Out[27]: foo bar
 a 0.865257 0.213169
 b 0.442759 0.108267
 c 0.047110 0.905718
 ```
(5) 通过 NumPy 结构化数组创建。2.9 节曾介绍过结构化数组。由于 Pandas 的 DataFrame
与结构化数组十分相似，因此可以通过结构化数组创建 DataFrame：
```python
In[28]: A = np.zeros(3, dtype=[('A', 'i8'), ('B', 'f8')])
 A
Out[28]: array([(0, 0.0), (0, 0.0), (0, 0.0)],
 dtype=[('A', '<i8'), ('B', '<f8')])
In[29]: pd.DataFrame(A)
Out[29]: A B
 0 0 0.0
 1 0 0.0
 2 0 0.0
 
### Pandas的Index对象
Series 和 DataFrame 对象都使用便于引用和调整的显式索引。Pandas 的
Index 对象是一个很有趣的数据结构，可以将它看作是一个不可变数组或有序集合（实际
上是一个多集，因为 Index 对象可能会包含重复值）。这两种观点使得 Index 对象能呈现一
些有趣的功能。让我们用一个简单的整数列表来创建一个 Index 对象：
```python
In[30]: ind = pd.Index([2, 3, 5, 7, 11])
 ind
Out[30]: Int64Index([2, 3, 5, 7, 11], dtype='int64')

#### 1. 将Index看作不可变数组
Index 对象的许多操作都像数组。例如，可以通过标准 Python 的取值方法获取数值，也可
以通过切片获取数值：
```python
In[31]: ind[1]
Out[31]: 3
In[32]: ind[::2]
Out[32]: Int64Index([2, 5, 11], dtype='int64')
```
Index 对象还有许多与 NumPy 数组相似的属性：
```python
In[33]: print(ind.size, ind.shape, ind.ndim, ind.dtype)
5 (5,) 1 int64
```
Index 对象与 NumPy 数组之间的不同在于，Index 对象的索引是不可变的，也就是说不能
通过通常的方式进行调整：
```python
In[34]: ind[1] = 0
---------------------------------------------------------------------------
TypeError Traceback (most recent call last)
<ipython-input-34-40e631c82e8a> in <module>()
----> 1 ind[1] = 0
/Users/jakevdp/anaconda/lib/python3.5/site-packages/pandas/indexes/base.py ...
 1243
 1244 def __setitem__(self, key, value):
-> 1245 raise TypeError("Index does not support mutable operations")
 1246
 1247 def __getitem__(self, key):
TypeError: Index does not support mutable operations
```
Index 对象的不可变特征使得多个 DataFrame 和数组之间进行索引共享时更加安全，尤其是
可以避免因修改索引时粗心大意而导致的副作用。
#### 2. 将Index看作有序集合
Pandas 对象被设计用于实现许多操作，如连接（join）数据集，其中会涉及许多集合操作。
Index 对象遵循 Python 标准库的集合（set）数据结构的许多习惯用法，包括并集、交集、
差集等：
```python
In[35]: indA = pd.Index([1, 3, 5, 7, 9])
 indB = pd.Index([2, 3, 5, 7, 11])
In[36]: indA & indB # 交集
Out[36]: Int64Index([3, 5, 7], dtype='int64')
In[37]: indA | indB # 并集


Out[37]: Int64Index([1, 2, 3, 5, 7, 9, 11], dtype='int64')
In[38]: indA ^ indB # 异或
Out[38]: Int64Index([1, 2, 9, 11], dtype='int64')
```
这些操作还可以通过调用对象方法来实现，例如 indA.intersection(indB)。
## 　数据取值与选择
第 2 章具体介绍了获取、设置、调整 NumPy 数组数值的方法与工具，包括取值操作（如
arr[2, 1]）、切片操作（如 arr[:, 1:5]）、掩码操作（如 arr[arr > 0]）、花哨的索引操作
（如 arr[0, [1, 5]]），以及组合操作（如 arr[:, [1, 5]]）。下面介绍 Pandas 的 Series 和
DataFrame 对象相似的数据获取与调整操作。如果你用过 NumPy 操作模式，就会非常熟悉
Pandas 的操作模式，只是有几个细节需要注意一下。
我们将从简单的一维 Series 对象开始，然后再用比较复杂的二维 DataFrame 对象进行
演示。
###  Series数据选择方法
如前所述，Series 对象与一维 NumPy 数组和标准 Python 字典在许多方面都一样。只要牢
牢记住这两个类比，就可以帮助我们更好地理解 Series 对象的数据索引与选择模式。
#### 1. 将Series看作字典
和字典一样，Series 对象提供了键值对的映射：
```python
In[1]: import pandas as pd
 data = pd.Series([0.25, 0.5, 0.75, 1.0],
 index=['a', 'b', 'c', 'd'])
 data
Out[1]: a 0.25
 b 0.50
 c 0.75
 d 1.00
 dtype: float64
In[2]: data['b']
Out[2]: 0.5
```
我们还可以用 Python 字典的表达式和方法来检测键 / 索引和值：
```python
In[3]: 'a' in data
Out[3]: True
In[4]: data.keys()

Out[4]: Index(['a', 'b', 'c', 'd'], dtype='object')
In[5]: list(data.items())
Out[5]: [('a', 0.25), ('b', 0.5), ('c', 0.75), ('d', 1.0)]
```
Series 对象还可以用字典语法调整数据。就像你可以通过增加新的键扩展字典一样，你也
可以通过增加新的索引值扩展 Series：
```python
In[6]: data['e'] = 1.25
 data
Out[6]: a 0.25
 b 0.50
 c 0.75
 d 1.00
 e 1.25
 dtype: float64
 ```
Series 对象的可变性是一个非常方便的特性：Pandas 在底层已经为可能发生的内存布局和
数据复制自动决策，用户不需要担心这些问题。
#### 2. 将Series看作一维数组
Series 不仅有着和字典一样的接口，而且还具备和 NumPy 数组一样的数组数据选择功能，
包括索引、掩码、花哨的索引等操作，具体示例如下所示：
```python
In[7]: # 将显式索引作为切片
 data['a':'c']
Out[7]: a 0.25
 b 0.50
 c 0.75
 dtype: float64
In[8]: # 将隐式整数索引作为切片
 data[0:2]
Out[8]: a 0.25
 b 0.50
 dtype: float64
In[9]: # 掩码
 data[(data > 0.3) & (data < 0.8)]
Out[9]: b 0.50
 c 0.75
 dtype: float64
In[10]: # 花哨的索引
 data[['a', 'e']]
Out[10]: a 0.25
 e 1.25
 dtype: float64
```
在以上示例中，切片是绝大部分混乱之源。需要注意的是，当使用显式索引（即
data['a':'c']）作切片时，结果包含最后一个索引；而当使用隐式索引（即 data[0:2]）
作切片时，结果不包含最后一个索引。
#### 3. 索引器：loc、iloc和ix
这些切片和取值的习惯用法经常会造成混乱。例如，如果你的 Series 是显式整数索引，那
么 data[1] 这样的取值操作会使用显式索引，而 data[1:3] 这样的切片操作却会使用隐式
索引。
```python
In[11]: data = pd.Series(['a', 'b', 'c'], index=[1, 3, 5])
 data
Out[11]: 1 a
 3 b
 5 c
 dtype: object
In[12]: # 取值操作是显式索引
 data[1]
Out[12]: 'a'
In[13]: # 切片操作是隐式索引
 data[1:3]
Out[13]: 3 b
 5 c
 dtype: object
 ```
由于整数索引很容易造成混淆，所以 Pandas 提供了一些索引器（indexer）属性来作为取值
的方法。它们不是 Series 对象的函数方法，而是暴露切片接口的属性。
第一种索引器是 loc 属性，表示取值和切片都是显式的：
```python
In[14]: data.loc[1]
Out[14]: 'a'
In[15]: data.loc[1:3]
Out[15]: 1 a
 3 b
 dtype: object
 ```
第二种是 iloc 属性，表示取值和切片都是 Python 形式的 1 隐式索引：
```python
In[16]: data.iloc[1]
Out[16]: 'b'
In[17]: data.iloc[1:3]


Out[17]: 3 b
 5 c
 dtype: object
 ```
 注 1：从 0 开始，左闭右开区间。——译者注
第三种取值属性是 ix，它是前两种索引器的混合形式，在 Series 对象中 ix 等价于标准的
[]（Python 列表）取值方式。ix 索引器主要用于 DataFrame 对象，后面将会介绍。
Python 代码的设计原则之一是“显式优于隐式”。使用 loc 和 iloc 可以让代码更容易维护，
可读性更高。特别是在处理整数索引的对象时，我强烈推荐使用这两种索引器。它们既可
以让代码阅读和理解起来更容易，也能避免因误用索引 / 切片而产生的小 bug。
###  DataFrame数据选择方法
前面曾提到，DataFrame 在有些方面像二维或结构化数组，在有些方面又像一个共享索引
的若干 Series 对象构成的字典。这两种类比可以帮助我们更好地掌握这种数据结构的数据
选择方法。
#### 1. 将DataFrame看作字典
第一种类比是把 DataFrame 当作一个由若干 Series 对象构成的字典。让我们用之前的美国
五州面积与人口数据来演示：
```python
In[18]: area = pd.Series({'California': 423967, 'Texas': 695662,
 'New York': 141297, 'Florida': 170312,
 'Illinois': 149995})
 pop = pd.Series({'California': 38332521, 'Texas': 26448193,
 'New York': 19651127, 'Florida': 19552860,
 'Illinois': 12882135})
 data = pd.DataFrame({'area':area, 'pop':pop})
 data
Out[18]: area pop
 California 423967 38332521
 Florida 170312 19552860
 Illinois 149995 12882135
 New York 141297 19651127
 Texas 695662 
 ```
两个 Series 分别构成 DataFrame 的
一列，可以通过对列名进行字典形式（dictionary-style）
的取值获取数据：
```python
In[19]: data['area']
Out[19]: California 423967
 Florida 170312
 Illinois 149995
 New York 141297
 Texas 695662
 Name: area, dtype: int64
 ```
同样，也可以用属性形式（attribute-style）选择纯字符串列名的数据：
```python
In[20]: data.area
Out[20]: California 423967
 Florida 170312
 Illinois 149995
 New York 141297
 Texas 695662
 Name: area, dtype: int64
 ```
对同一个对象进行属性形式与字典形式的列数据，结果是相同的：
```python
In[21]: data.area is data['area']
Out[21]: True
```
虽然属性形式的数据选择方法很方便，但是它并不是通用的。如果列名不是纯字符串，或
者列名与 DataFrame 的方法同名，那么就不能用属性索引。例如，DataFrame 有一个 pop()
方法，如果用 data.pop 就不会获取 'pop' 列，而是显示为方法：
```python
In[22]: data.pop is data['pop']
Out[22]: False
```
另外，还应该避免对用属性形式选择的列直接赋值（即可以用 data['pop'] = z，但不要用
data.pop = z）。
和前面介绍的 Series 对象一样，还可以用字典形式的语法调整对象，如果要增加一列可以
这样做：
```python
In[23]: data['density'] = data['pop'] / data['area']
 data
Out[23]: area pop density
 California 423967 38332521 90.413926
 Florida 170312 19552860 114.806121
 Illinois 149995 12882135 85.883763
 New York 141297 19651127 139.076746
 Texas 695662 26448193 38.018740
 ```
这里演示了两个 Series 对象算术运算的简便语法，我们将在 3.4 节进行详细介绍。
#### 2. 将DataFrame看作二维数组
前面曾提到，可以把 DataFrame 看成是一个增强版的二维数组，用 values 属性按行查看数
组数据：
```python
In[24]: data.values
Out[24]: array([[ 4.23967000e+05, 3.83325210e+07, 9.04139261e+01],
 [ 1.70312000e+05, 1.95528600e+07, 1.14806121e+02],
 [ 1.49995000e+05, 1.28821350e+07, 8.58837628e+01],
 [ 1.41297000e+05, 1.96511270e+07, 1.39076746e+02],
 [ 6.95662000e+05, 2.64481930e+07, 3.80187404e+01]])
 ```
理解了这一点，就可以把许多数组操作方式用在 DataFrame 上。例如，可以对 DataFrame

进行行列转置：
```python
In[25]: data.T
Out[25]:
 California Florida Illinois New York Texas
area 4.239670e+05 1.703120e+05 1.499950e+05 1.412970e+05 6.956620e+05
pop 3.833252e+07 1.955286e+07 1.288214e+07 1.965113e+07 2.644819e+07
density 9.041393e+01 1.148061e+02 8.588376e+01 1.390767e+02 3.801874e+01
```
通过字典形式对列进行取值显然会限制我们把 DataFrame 作为 NumPy 数组可以获得的能
力，尤其是当我们在 DataFrame 数组中使用单个行索引获取一行数据时：
```python
In[26]: data.values[0]
Out[26]: array([ 4.23967000e+05, 3.83325210e+07, 9.04139261e+01])

而获取一列数据就需要向 DataFrame 传递单个列索引：
```python
In[27]: data['area']
Out[27]: California 423967
 Florida 170312
 Illinois 149995
 New York 141297
 Texas 695662
 Name: area, dtype: int64
 ```
因此，在进行数组形式的取值时，我们就需要用另一种方法——前面介绍过的 Pandas 索引
器 loc、iloc 和 ix 了。通过 iloc 索引器，我们就可以像对待 NumPy 数组一样索引 Pandas
的底层数组（Python 的隐式索引），DataFrame 的行列标签会自动保留在结果中：
```python
In[28]: data.iloc[:3, :2]
Out[28]: area pop
 California 423967 38332521
 Florida 170312 19552860
 Illinois 149995 12882135
In[29]: data.loc[:'Illinois', :'pop']
Out[29]: area pop
 California 423967 38332521
 Florida 170312 19552860
 Illinois 149995 12882135
 ```
使用 ix 索引器可以实现一种混合效果：
```python
In[30]: data.ix[:3, :'pop']
Out[30]: area pop
 California 423967 38332521
 Florida 170312 19552860
 Illinois 149995 12882135
 ```
需要注意的是，ix 索引器对于整数索引的处理和之前在 Series 对象中介绍的一样，都容

易让人混淆。
任何用于处理 NumPy 形式数据的方法都可以用于这些索引器。例如，可以在 loc 索引器
中结合使用掩码与花哨的索引方法：
```python
In[31]: data.loc[data.density > 100, ['pop', 'density']]
Out[31]: pop density
 Florida 19552860 114.806121
 New York 19651127 139.076746
 ```
任何一种取值方法都可以用于调整数据，这一点和 NumPy 的常用方法是相同的：
```python
In[32]: data.iloc[0, 2] = 90
 data
Out[32]: area pop density
 California 423967 38332521 90.000000
 Florida 170312 19552860 114.806121
 Illinois 149995 12882135 85.883763
 New York 141297 19651127 139.076746
 Texas 695662 26448193 38.018740
 ```
如果你想熟练使用 Pandas 的数据操作方法，我建议你花点时间在一个简单的 DataFrame 上
练习不同的取值方法，包括查看索引类型、切片、掩码和花哨的索引操作。
#### 3. 其他取值方法
还有一些取值方法和前面介绍过的方法不太一样。它们虽然看着有点奇怪，但是在实践中
还是很好用的。首先，如果对单个标签取值就选择列，而对多个标签用切片就选择行：
```python
In[33]: data['Florida':'Illinois']
Out[33]: area pop density
 Florida 170312 19552860 114.806121
 Illinois 149995 12882135 85.883763
 ```
切片也可以不用索引值，而直接用行数来实现：
```python
In[34]: data[1:3]
Out[34]: area pop density
 Florida 170312 19552860 114.806121
 Illinois 149995 12882135 85.883763
 ```
与之类似，掩码操作也可以直接对每一行进行过滤，而不需要使用 loc 索引器：
```python
In[35]: data[data.density > 100]
Out[35]: area pop density
 Florida 170312 19552860 114.806121
 New York 141297 19651127 139.076746
 ```
这两种操作方法其实与 NumPy 数组的语法类似，虽然它们与 Pandas 的操作习惯不太一致，
但是在实践中非常好用。

## Pandas数值运算方法
NumPy 的基本能力之一是快速对每个元素进行运算，既包括基本算术运算（加、减、乘、
除），也包括更复杂的运算（三角函数、指数函数和对数函数等）。Pandas 继承了 NumPy
的功能，在 2.3 节介绍过的通用函数是关键。
但是 Pandas 也实现了一些高效技巧：对于一元运算（像函数与三角函数），这些通用函
数将在输出结果中保留索引和列标签；而对于二元运算（如加法和乘法），Pandas 在传递
通用函数时会自动对齐索引进行计算。这就意味着，保存数据内容与组合不同来源的数
据——两处在 NumPy 数组中都容易出错的地方——变成了 Pandas 的杀手锏。后面还会介
绍一些关于一维 Series 和二维 DataFrame 的便捷运算方法。
###　通用函数：保留索引
因为 Pandas 是建立在 NumPy 基础之上的，所以 NumPy 的通用函数同样适用于 Pandas 的
Series 和 DataFrame 对象。让我们用一个简单的 Series 和 DataFrame 来演示：
```python
In[1]: import pandas as pd
 import numpy as np
In[2]: rng = np.random.RandomState(42)
 ser = pd.Series(rng.randint(0, 10, 4))
 ser
Out[2]: 0 6
 1 3
 2 7
 3 4
 dtype: int64
In[3]: df = pd.DataFrame(rng.randint(0, 10, (3, 4)),
 columns=['A', 'B', 'C', 'D'])
 df
Out[3]: A B C D
 0 6 9 2 6
 1 7 4 3 7
 2 7 2 5 4
 ```
如果对这两个对象的其中一个使用 NumPy 通用函数，生成的结果是另一个保留索引的
Pandas 对象：
```python
In[4]: np.exp(ser)
Out[4]: 0 403.428793
 1 20.085537
 2 1096.633158
 3 54.598150
 dtype: float64
 ```
或者，再做一个比较复杂的运算：
```python
In[5]: np.sin(df * np.pi / 4)
Out[5]: A B C D
 0 -1.000000 7.071068e-01 1.000000 -1.000000e+00
 1 -0.707107 1.224647e-16 0.707107 -7.071068e-01
 2 -0.707107 1.000000e+00 -0.707107 1.224647e-16
 ```
任何一种在 2.3 节介绍过的通用函数都可以按照类似的方式使用。
### 　通用函数：索引对齐
当在两个 Series 或 DataFrame 对象上进行二元计算时，Pandas 会在计算过程中对齐两个对
象的索引。当你处理不完整的数据时，这一点非常方便，我们将在后面的示例中看到。
#### 1. Series索引对齐
来看一个例子，假如你要整合两个数据源的数据，其中一个是美国面积最大的三个州的面
积数据，另一个是美国人口最多的三个州的人口数据：
```python
In[6]: area = pd.Series({'Alaska': 1723337, 'Texas': 695662,
 'California': 423967}, name='area')
 population = pd.Series({'California': 38332521, 'Texas': 26448193,
 'New York': 19651127}, name='population')
 ```
来看看如果用人口除以面积会得到什么样的结果：
```python
In[7]: population / area
Out[7]: Alaska NaN
 California 90.413926
 New York NaN
 Texas 38.018740
 dtype: float64
 ```
结果数组的索引是两个输入数组索引的并集。我们也可以用 Python 标准库的集合运算法则
来获得这个索引：
```python
In[8]: area.index | population.index
Out[8]: Index(['Alaska', 'California', 'New York', 'Texas'], dtype='object')
```
对于缺失位置的数据，Pandas 会用 NaN 填充，表示“此处无数”。这是 Pandas 表示缺失值
的方法（详情请参见 3.5 节关于缺失值的介绍）。这种索引对齐方式是通过 Python 内置的
集合运算规则实现的，任何缺失值默认都用 NaN 填充：
```python
In[9]: A = pd.Series([2, 4, 6], index=[0, 1, 2])
 B = pd.Series([1, 3, 5], index=[1, 2, 3])
 A + B
Out[9]: 0 NaN
 1 5.0
 2 9.0
 3 NaN
 dtype: float64
```
如果用 NaN 值不是我们想要的结果，那么可以用适当的对象方法代替运算符。例如，
A.add(B) 等价于 A + B，也可以设置参数自定义 A 或 B 缺失的数据：
```python
In[10]: A.add(B, fill_value=0)
Out[10]: 0 2.0
 1 5.0
 2 9.0
 3 5.0
 dtype: float64
 ```
#### 2. DataFrame索引对齐
在计算两个 DataFrame 时，类似的索引对齐规则也同样会出现在共同（并集）列中：
```python
In[11]: A = pd.DataFrame(rng.randint(0, 20, (2, 2)),
 columns=list('AB'))
 A
Out[11]: A B
 0 1 11
 1 5 1
In[12]: B = pd.DataFrame(rng.randint(0, 10, (3, 3)),
 columns=list('BAC'))
 B
Out[12]: B A C
 0 4 0 9
 1 5 8 0
 2 9 2 6
In[13]: A + B
Out[13]: A B C
 0 1.0 15.0 NaN
 1 13.0 6.0 NaN
 2 NaN NaN NaN
 ```
你会发现，两个对象的行列索引可以是不同顺序的，结果的索引会自动按顺序排列。在
Series 中，我们可以通过运算符方法的 fill_value 参数自定义缺失值。这里，我们将用 A
中所有值的均值来填充缺失值（计算 A 的均值需要用 stack 将二维数组压缩成一维数组）：
```python
In[14]: fill = A.stack().mean()
 A.add(B, fill_value=fill)
Out[14]: A B C
 0 1.0 15.0 13.5
 1 13.0 6.0 4.5
 2 6.5 13.5 10.5
 ```
表 3-1 列举了与 Python 运算符相对应的 Pandas 对象方法。

表3-1：Python运算符与Pandas方法的映射关系
Python运算符 Pandas方法
+ add()
- sub()、subtract()
* mul()、multiply()
/ truediv()、div()、divide()
// floordiv()
% mod()
** pow()

### 　通用函数：DataFrame与Series的运算
我们经常需要对一个 DataFrame 和一个 Series 进行计算，行列对齐方式与之前类似。也就
是说，DataFrame 和 Series 的运算规则，与 NumPy 中二维数组与一维数组的运算规则是
一样的。来看一个常见运算，让一个二维数组减去自身的一行数据：
```python
In[15]: A = rng.randint(10, size=(3, 4))
 A
Out[15]: array([[3, 8, 2, 4],
 [2, 6, 4, 8],
 [6, 1, 3, 8]])
In[16]: A - A[0]
Out[16]: array([[ 0, 0, 0, 0],
 [-1, -2, 2, 4],
 [ 3, -7, 1, 4]])
 ```
根据 NumPy 的广播规则（详情请参见 2.5 节），让二维数组减自身的一行数据会按行计算。
在 Pandas 里默认也是按行运算的：
```python
In[17]: df = pd.DataFrame(A, columns=list('QRST'))
 df - df.iloc[0]
Out[17]: Q R S T
 0 0 0 0 0
 1 -1 -2 2 4
 2 3 -7 1 4
 ```
如果你想按列计算，那么就需要利用前面介绍过的运算符方法，通过 axis 参数设置：
```python
In[18]: df.subtract(df['R'], axis=0)
Out[18]: Q R S T
 0 -5 0 -6 -4
 1 -4 0 -2 2
 2 5 0 2 7
```
你会发现 DataFrame / Series 的运算与前面介绍的运算一样，结果的索引都会自动对齐：
```python
In[19]: halfrow = df.iloc[0, ::2]
 halfrow
Out[19]: Q 3
 S 2
 Name: 0, dtype: int64
In[20]: df - halfrow
Out[20]: Q R S T
 0 0.0 NaN 0.0 NaN
 1 -1.0 NaN 2.0 NaN
 2 3.0 NaN 1.0 NaN
 ```
这些行列索引的保留与对齐方法说明 Pandas 在运算时会一直保存这些数据内容，从而避免
在处理数据类型有差异和 / 或维度不一致的 NumPy 数组时可能遇到的问题。
## 　处理缺失值
大多数教程里使用的数据与现实工作中的数据的区别在于后者很少是干净整齐的，许多
目前流行的数据集都会有数据缺失的现象。更为甚者，处理不同数据源缺失值的方法还
不同。
我们将在本节介绍一些处理缺失值的通用规则，Pandas 对缺失值的表现形式，并演示
Pandas 自带的几个处理缺失值的工具的用法。本节以及全书涉及的缺失值主要有三种形
式：null、NaN 或 NA。
###　选择处理缺失值的方法
在数据表或 DataFrame 中有很多识别缺失值的方法。一般情况下可以分为两种：  
一种方法是通过一个覆盖全局的掩码表示缺失值，  
另一种方法是用一个标签值（sentinel value）表示缺失值。  
在掩码方法中，掩码可能是一个与原数组维度相同的完整布尔类型数组，也可能是用一个
比特（0 或 1）表示有缺失值的局部状态。  
在标签方法中，标签值可能是具体的数据（例如用 -9999 表示缺失的整数），也可能是些
极少出现的形式。另外，标签值还可能是更全局的值，比如用 NaN（不是一个数）表示缺
失的浮点数，它是 IEEE 浮点数规范中指定的特殊字符。
使用这两种方法之前都需要先综合考量：使用单独的掩码数组会额外出现一个布尔类型数
组，从而增加存储与计算的负担；而标签值方法缩小了可以被表示为有效值的范围，可能
需要在 CPU 或 GPU 算术逻辑单元中增加额外的（往往也不是最优的）计算逻辑。通常使
用的 NaN 也不能表示所有数据类型。
大多数情况下，都不存在最佳选择，不同的编程语言与系统使用不同的方法。例如，R 语
言在每种数据类型中保留一个比特作为缺失数据的标签值，而 SciDB 系统会在每个单元后
面加一个额外的字节表示 NA 状态。

### Pandas的缺失值
Pandas 里处理缺失值的方式延续了 NumPy 程序包的方式，并没有为浮点数据类型提供内
置的 NA 作为缺失值。
Pandas 原本也可以按照 R 语言采用的比特模式为每一种数据类型标注缺失值，但是这种方
法非常笨拙。R 语言包含 4 种基本数据类型，而 NumPy 支持的类型远超 4 种。例如，R 语
言只有一种整数类型，而 NumPy 支持 14 种基本的整数类型，可以根据精度、符号、编码
类型按需选择。如果要为 NumPy 的每种数据类型都设置一个比特标注缺失值，可能需要
为不同类型的不同操作耗费大量的时间与精力，其工作量几乎相当于创建一个新的 NumPy
程序包。另外，对于一些较小的数据类型（例如 8 位整型数据），牺牲一个比特作为缺失
值标注的掩码还会导致其数据范围缩小。  
当然，NumPy 也是支持掩码数据的，也就是说可以用一个布尔掩码数组为原数组标注“无
缺失值”或“有缺失值”。Pandas 也集成了这个功能，但是在存储、计算和编码维护方面
都需要耗费不必要的资源，因此这种方式并不可取。  
综合考虑各种方法的优缺点，Pandas 最终选择用标签方法表示缺失值，包括两种 Python 原
有的缺失值：浮点数据类型的 NaN 值，以及 Python 的 None 对象。后面我们将会发现，虽
然这么做也会有一些副作用，但是在实际运用中的效果还是不错的。
####1. None：Python对象类型的缺失值
Pandas 可以使用的第一种缺失值标签是 None，它是一个 Python 单体对象，经常在代码中
表示缺失值。由于 None 是一个 Python 对象，所以不能作为任何 NumPy / Pandas 数组类型
的缺失值，只能用于 'object' 数组类型（即由 Python 对象构成的数组）：
```python
In[1]: import numpy as np
 import pandas as pd
In[2]: vals1 = np.array([1, None, 3, 4])
 vals1
Out[2]: array([1, None, 3, 4], dtype=object)
```
这里 dtype=object 表示 NumPy 认为由于这个数组是 Python 对象构成的，因此将其类型
判断为 object。虽然这种类型在某些情景中非常有用，对数据的任何操作最终都会在
Python 层面完成，但是在进行常见的快速操作时，这种类型比其他原生类型数组要消耗
更多的资源：
```python
In[3]: for dtype in ['object', 'int']:
 print("dtype =", dtype)
 %timeit np.arange(1E6, dtype=dtype).sum()
 print()
dtype = object
10 loops, best of 3: 78.2 ms per loop
dtype = int
100 loops, best of 3: 3.06 ms per loop
```
使用 Python 对象构成的数组就意味着如果你对一个包含 None 的数组进行累计操作，如
sum() 或者 min()，那么通常会出现类型错误：
```python
In[4]: vals1.sum()
TypeError Traceback (most recent call last)
<ipython-input-4-749fd8ae6030> in <module>()
----> 1 vals1.sum()
/Users/jakevdp/anaconda/lib/python3.5/site-packages/numpy/core/_methods.py ...
 30
 31 def _sum(a, axis=None, dtype=None, out=None, keepdims=False):
---> 32 return umr_sum(a, axis, dtype, out, keepdims)
 33
 34 def _prod(a, axis=None, dtype=None, out=None, keepdims=False):
TypeError: unsupported operand type(s) for +: 'int' and 'NoneType'
```
这就是说，在 Python 中没有定义整数与 None 之间的加法运算。
#### 2. NaN：数值类型的缺失值
另一种缺失值的标签是 NaN（全称 Not a Number，不是一个数字），是一种按照 IEEE 浮点
数标准设计、在任何系统中都兼容的特殊浮点数：
```python
In[5]: vals2 = np.array([1, np.nan, 3, 4])
 vals2.dtype
Out[5]: dtype('float64')
```
请注意，NumPy 会为这个数组选择一个原生浮点类型，这意味着和之前的 object 类型数
组不同，这个数组会被编译成 C 代码从而实现快速操作。你可以把 NaN 看作是一个数据类
病毒——它会将与它接触过的数据同化。无论和 NaN 进行何种操作，最终结果都是 NaN：
```python
In[6]: 1 + np.nan
Out[6]: nan
In[7]: 0 * np.nan
Out[7]: nan
```
虽然这些累计操作的结果定义是合理的（即不会抛出异常），但是并非总是有效的：
```python
In[8]: vals2.sum(), vals2.min(), vals2.max()
Out[8]: (nan, nan, nan)
```
NumPy 也提供了一些特殊的累计函数，它们可以忽略缺失值的影响：
```python
In[9]: np.nansum(vals2), np.nanmin(vals2), np.nanmax(vals2)
Out[9]: (8.0, 1.0, 4.0)
```
谨记，NaN 是一种特殊的浮点数，不是整数、字符串以及其他数据类型。
#### 3. Pandas中NaN与None的差异
虽然 NaN 与 None 各有各的用处，但是 Pandas 把它们看成是可以等价交换的，在适当的时
候会将两者进行替换：
```python
In[10]: pd.Series([1, np.nan, 2, None])
Out[10]: 0 1.0
 1 NaN
 2 2.0
 3 NaN
 dtype: float64
 ```
Pandas 会将没有标签值的数据类型自动转换为 NA。例如，当我们将整型数组中的一个值设
置为 np.nan 时，这个值就会强制转换成浮点数缺失值 NA。
```python
In[11]: x = pd.Series(range(2), dtype=int)
 x
Out[11]: 0 0
 1 1
 dtype: int64
In[12]: x[0] = None
 x
Out[12]: 0 NaN
 1 1.0
 dtype: float64
 ```
请注意，除了将整型数组的缺失值强制转换为浮点数，Pandas 还会自动将 None 转换为
NaN。（需要注意的是，现在 GitHub 上 Pandas 项目中已经有人提议增加一个原生的整型
NA，不过到编写本书时还尚未实现。）
尽管这些仿佛会魔法的类型比 R 语言等专用统计语言的缺失值要复杂一些，但是 Pandas
的标签 / 转换方法在实践中的效果非常好，在我个人的使用过程中几乎没有出过问题。
Pandas 对 NA 缺失值进行强制转换的规则如表 3-2 所示。
表3-2：Pandas对不同类型缺失值的转换规则
类型 缺失值转换规则 NA标签值
floating 浮点型 无变化 np.nan
object 对象类型 无变化 None 或 np.nan
integer 整数类型 强制转换为 float64 np.nan
boolean 布尔类型 强制转换为 object None 或 np.nan
需要注意的是，Pandas 中字符串类型的数据通常是用 object 类型存储的。

###　处理缺失值
我们已经知道，Pandas 基本上把 None 和 NaN 看成是可以等价交换的缺失值形式。为了完成
这种交换过程，Pandas 提供了一些方法来发现、剔除、替换数据结构中的缺失值，主要包
括以下几种。
isnull()
创建一个布尔类型的掩码标签缺失值。
notnull()
与 isnull() 操作相反。
dropna()
返回一个剔除缺失值的数据。
fillna()
返回一个填充了缺失值的数据副本。
本节将用简单的示例演示这些方法。
#### 1. 发现缺失值
Pandas 数据结构有两种有效的方法可以发现缺失值：isnull() 和 notnull()。每种方法都
返回布尔类型的掩码数据，例如：
```python
In[13]: data = pd.Series([1, np.nan, 'hello', None])
In[14]: data.isnull()
Out[14]: 0 False
 1 True
 2 False
 3 True
 dtype: bool
 ```
就像在 3.3 节中介绍的，布尔类型掩码数组可以直接作为 Series 或 DataFrame 的索引使用：
```python
In[15]: data[data.notnull()]
Out[15]: 0 1
 2 hello
 dtype: object
 ```
在 Series 里使用的 isnull() 和 notnull() 同样适用于 DataFrame，产生的结果同样是布尔
类型。
#### 2. 剔除缺失值
除了前面介绍的掩码方法，还有两种很好用的缺失值处理方法，分别是 dropna()（剔除缺
失值）和 fillna()（填充缺失值）。在 Series 上使用这些方法非常简单：
```python
In[16]: data.dropna()
Out[16]: 0 1 

 2 hello
 dtype: object
 ```
而在 DataFrame 上使用它们时需要设置一些参数，例如下面的 DataFrame：
```python
In[17]: df = pd.DataFrame([[1, np.nan, 2],
 [2, 3, 5],
 [np.nan, 4, 6]])
 df
Out[17]: 0 1 2
 0 1.0 NaN 2
 1 2.0 3.0 5
 2 NaN 4.0 6
 ```
我们没法从 DataFrame 中单独剔除一个值，要么是剔除缺失值所在的整行，要么是整列。
根据实际需求，有时你需要剔除整行，有时可能是整列，DataFrame 中的 dropna() 会有一
些参数可以配置。
默认情况下，dropna() 会剔除任何包含缺失值的整行数据：
```python
In[18]: df.dropna()
Out[18]: 0 1 2
 1 2.0 3.0 5
 ```
可以设置按不同的坐标轴剔除缺失值，比如 axis=1（或 axis='columns'）会剔除任何包含
缺失值的整列数据：
```python
In[19]: df.dropna(axis='columns')
Out[19]: 2
 0 2
 1 5
 2 6
 ```
但是这么做也会把非缺失值一并剔除，因为可能有时候只需要剔除全部是缺失值的行或
列，或者绝大多数是缺失值的行或列。这些需求可以通过设置 how 或 thresh 参数来满足，
它们可以设置剔除行或列缺失值的数量阈值。
默认设置是 how='any'，也就是说只要有缺失值就剔除整行或整列（通过 axis 设置坐标
轴）。你还可以设置 how='all'，这样就只会剔除全部是缺失值的行或列了：
```python
In[20]: df[3] = np.nan
 df
Out[20]: 0 1 2 3
 0 1.0 NaN 2 NaN
 1 2.0 3.0 5 NaN
 2 NaN 4.0 6 NaN
In[21]: df.dropna(axis='columns', how='all')
Out[21]: 0 1 2

 0 1.0 NaN 2
 1 2.0 3.0 5
 2 NaN 4.0 6
 ```
还可以通过 thresh 参数设置行或列中非缺失值的最小数量，从而实现更加个性化的配置：
```python
In[22]: df.dropna(axis='rows', thresh=3)
Out[22]: 0 1 2 3
 1 2.0 3.0 5 NaN
 ```
第 1 行与第 3 行被剔除了，因为它们只包含两个非缺失值。
#### 3. 填充缺失值
有时候你可能并不想移除缺失值，而是想把它们替换成有效的数值。有效的值可能是像
0、1、2 那样单独的值，也可能是经过填充（imputation）或转换（interpolation）得到的。
虽然你可以通过 isnull() 方法建立掩码来填充缺失值，但是 Pandas 为此专门提供了一个
fillna() 方法，它将返回填充了缺失值后的数组副本。
来用下面的 Series 演示：
```python
In[23]: data = pd.Series([1, np.nan, 2, None, 3], index=list('abcde'))
 data
Out[23]: a 1.0
 b NaN
 c 2.0
 d NaN
 e 3.0
 dtype: float64
 ```
我们将用一个单独的值来填充缺失值，例如用 0：
```python
In[24]: data.fillna(0)
Out[24]: a 1.0
 b 0.0
 c 2.0
 d 0.0
 e 3.0
 dtype: float64
 ```
可以用缺失值前面的有效值来从前往后填充（forward-fill）：
```python
In[25]: # 从前往后填充
 data.fillna(method='ffill')
Out[25]: a 1.0
 b 1.0
 c 2.0
 d 2.0
 e 3.0
 dtype: float64
 ```
也可以用缺失值后面的有效值来从后往前填充（back-fill）：
```python
In[26]: # 从后往前填充
 data.fillna(method='bfill')
Out[26]: a 1.0
 b 2.0
 c 2.0
 d 3.0
 e 3.0
 dtype: float64
 ```
DataFrame 的操作方法与 Series 类似，只是在填充时需要设置坐标轴参数 axis：
```python
In[27]: df
Out[27]: 0 1 2 3
 0 1.0 NaN 2 NaN
 1 2.0 3.0 5 NaN
 2 NaN 4.0 6 NaN
In[28]: df.fillna(method='ffill', axis=1)
Out[28]: 0 1 2 3
 0 1.0 1.0 2.0 2.0
 1 2.0 3.0 5.0 5.0
 2 NaN 4.0 6.0 6.0
 ```
需要注意的是，假如在从前往后填充时，需要填充的缺失值前面没有值，那么它就仍然是
缺失值。
## 　层级索引
当目前为止，我们接触的都是一维数据和二维数据，用 Pandas 的 Series 和 DataFrame 对
象就可以存储。但我们也经常会遇到存储多维数据的需求，数据索引超过一两个键。因
此，Pandas 提供了 Panel 和 Panel4D 对象解决三维数据与四维数据（详情请参见 3.7 节）。
而在实践中，更直观的形式是通过层级索引（hierarchical indexing，也被称为多级索引，
multi-indexing）配合多个有不同等级（level）的一级索引一起使用，这样就可以将高维数
组转换成类似一维 Series 和二维 DataFrame 对象的形式。
在这一节中，我们将介绍创建 MultiIndex 对象的方法，多级索引数据的取值、切片和统计
值的计算，以及普通索引与层级索引的转换方法。
首先导入 Pandas 和 NumPy：
```python
In[1]: import pandas as pd
 import numpy as np
 ```
 ### 　多级索引Series
让我们看看如何用一维的 Series 对象表示二维数据——用一系列包含特征与数值的数据点
来简单演示。

#### 1. 笨办法
假设你想要分析美国各州在两个不同年份的数据。如果你用前面介绍的 Pandas 工具来处
理，那么可能会用一个 Python 元组来表示索引：
```python
In[2]: index = [('California', 2000), ('California', 2010),
 ('New York', 2000), ('New York', 2010),
 ('Texas', 2000), ('Texas', 2010)]
 populations = [33871648, 37253956,
 18976457, 19378102,
 20851820, 25145561]
 pop = pd.Series(populations, index=index)
 pop
Out[2]: (California, 2000) 33871648
 (California, 2010) 37253956
 (New York, 2000) 18976457
 (New York, 2010) 19378102
 (Texas, 2000) 20851820
 (Texas, 2010) 25145561
 dtype: int64
 ```
通过元组构成的多级索引，你可以直接在 Series 上取值或用切片查询数据：
```python
In[3]: pop[('California', 2010):('Texas', 2000)]
Out[3]: (California, 2010) 37253956
 (New York, 2000) 18976457
 (New York, 2010) 19378102
 (Texas, 2000) 20851820
 dtype: int64
 ```
但是这么做很不方便。假如你想要选择所有 2000 年的数据，那么就得用一些比较复杂的
（可能也比较慢的）清理方法了：
```python
In[4]: pop[[i for i in pop.index if i[1] == 2010]]
Out[4]: (California, 2010) 37253956
 (New York, 2010) 19378102
 (Texas, 2010) 25145561
 dtype: int64
 ```
这么做虽然也能得到需要的结果，但是与 Pandas 令人爱不释手的切片语法相比，这种方法
确实不够简洁（在处理较大的数据时也不够高效）。
#### 2. 好办法：Pandas多级索引
好在 Pandas 提供了更好的解决方案。用元组表示索引其实是多级索引的基础，Pandas
的 MultiIndex 类型提供了更丰富的操作方法。我们可以用元组创建一个多级索引，如
下所示：
```python
In[5]: index = pd.MultiIndex.from_tuples(index)
 index
Out[5]: MultiIndex(levels=[['California', 'New York', 'Texas'], [2000, 2010]],
 labels=[[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]])
```
你会发现 MultiIndex 里面有一个 levels 属性表示索引的等级——这样做可以将州名和年
份作为每个数据点的不同标签。
如果将前面创建的 pop 的索引重置（reindex）为 MultiIndex，就会看到层级索引：
```python
In[6]: pop = pop.reindex(index)
 pop
Out[6]: California 2000 33871648
 2010 37253956
 New York 2000 18976457
 2010 19378102
 Texas 2000 20851820
 2010 25145561
 dtype: int64
 ```
其中前两列表示 Series 的多级索引值，第三列是数据。你会发现有些行仿佛缺失了第一列
数据——这其实是多级索引的表现形式，每个空格与上面的索引相同。
现在可以直接用第二个索引获取 2010 年的全部数据，与 Pandas 的切片查询用法一致：
```python
In[7]: pop[:, 2010]
Out[7]: California 37253956
 New York 19378102
 Texas 25145561
 dtype: int64
 ```
结果是单索引的数组，正是我们需要的。与之前的元组索引相比，多级索引的语法更简
洁。（操作也更方便！）下面继续介绍层级索引的取值操作方法。
#### 3. 高维数据的多级索引
你可能已经注意到，我们其实完全可以用一个带行列索引的简单 DataFrame 代替前面的多
级索引。其实 Pandas 已经实现了类似的功能。unstack() 方法可以快速将一个多级索引的
Series 转化为普通索引的 DataFrame：
```python
In[8]: pop_df = pop.unstack()
 pop_df
Out[8]: 2000 2010
 California 33871648 37253956
 New York 18976457 19378102
 Texas 20851820 25145561
 ```
当然了，也有 stack() 方法实现相反的效果：
```python
In[9]: pop_df.stack()
Out[9]: California 2000 33871648
 2010 37253956
 New York 2000 18976457
 2010 19378102
 Texas 2000 20851820
 2010 25145561
 dtype: int64
```
你可能会纠结于为什么要费时间研究层级索引。其实理由很简单：如果我们可以用含多级
索引的一维 Series 数据表示二维数据，那么我们就可以用 Series 或 DataFrame 表示三维
甚至更高维度的数据。多级索引每增加一级，就表示数据增加一维，利用这一特点就可以
轻松表示任意维度的数据了。假如要增加一列显示每一年各州的人口统计指标（例如 18
岁以下的人口），那么对于这种带有 MultiIndex 的对象，增加一列就像 DataFrame 的操作
一样简单：
```python
In[10]: pop_df = pd.DataFrame({'total': pop,
 'under18': [9267089, 9284094,
 4687374, 4318033,
 5906301, 6879014]})
 pop_df
Out[10]: total under18
 California 2000 33871648 9267089
 2010 37253956 9284094
 New York 2000 18976457 4687374
 2010 19378102 4318033
 Texas 2000 20851820 5906301
 2010 25145561 6879014
 ```
另外，所有在 3.4 节介绍过的通用函数和其他功能也同样适用于层级索引。我们可以计算
上面数据中 18 岁以下的人口占总人口的比例：
```python
In[11]: f_u18 = pop_df['under18'] / pop_df['total']
 f_u18.unstack()
Out[11]: 2000 2010
 California 0.273594 0.249211
 New York 0.247010 0.222831
 Texas 0.283251 0.273568
 ```
同样，我们也可以快速浏览和操作高维数据。
3.6.2　多级索引的创建方法
为 Series 或 DataFrame 创建多级索引最直接的办法就是将 index 参数设置为至少二维的索
引数组，如下所示：
```python
In[12]: df = pd.DataFrame(np.random.rand(4, 2),
 index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
 columns=['data1', 'data2'])
 df
Out[12]: data1 data2
 a 1 0.554233 0.356072
 2 0.925244 0.219474
 b 1 0.441759 0.610054
 2 0.171495 0.886688
 ```
MultiIndex 的创建工作将在后台完成。

同理，如果你把将元组作为键的字典传递给 Pandas， Pandas 也会默认转换为 MultiIndex：
```python
In[13]: data = {('California', 2000): 33871648,
 ('California', 2010): 37253956,
 ('Texas', 2000): 20851820,
 ('Texas', 2010): 25145561,
 ('New York', 2000): 18976457,
 ('New York', 2010): 19378102}
 pd.Series(data)
Out[13]: California 2000 33871648
 2010 37253956
 New York 2000 18976457
 2010 19378102
 Texas 2000 20851820
 2010 25145561
 dtype: int64
 ```
但是有时候显式地创建 MultiIndex 也是很有用的，下面来介绍一些创建方法。
#### 1. 显式地创建多级索引
你可以用 pd.MultiIndex 中的类方法更加灵活地构建多级索引。例如，就像前面介绍的，
你可以通过一个有不同等级的若干简单数组组成的列表来构建 MultiIndex：
```python
In[14]: pd.MultiIndex.from_arrays([['a', 'a', 'b', 'b'], [1, 2, 1, 2]])
Out[14]: MultiIndex(levels=[['a', 'b'], [1, 2]],
 labels=[[0, 0, 1, 1], [0, 1, 0, 1]])
 ```
也可以通过包含多个索引值的元组构成的列表创建 MultiIndex：
```python
In[15]: pd.MultiIndex.from_tuples([('a', 1), ('a', 2), ('b', 1), ('b', 2)])
Out[15]: MultiIndex(levels=[['a', 'b'], [1, 2]],
 labels=[[0, 0, 1, 1], [0, 1, 0, 1]])
 ```
还可以用两个索引的笛卡尔积（Cartesian product）创建 MultiIndex：
```python
In[16]: pd.MultiIndex.from_product([['a', 'b'], [1, 2]])
Out[16]: MultiIndex(levels=[['a', 'b'], [1, 2]],
 labels=[[0, 0, 1, 1], [0, 1, 0, 1]])
 ```
更可以直接提供 levels（包含每个等级的索引值列表的列表）和 labels（包含每个索引值
标签列表的列表）创建 MultiIndex：
```python
In[17]: pd.MultiIndex(levels=[['a', 'b'], [1, 2]],
 labels=[[0, 0, 1, 1], [0, 1, 0, 1]])
Out[17]: MultiIndex(levels=[['a', 'b'], [1, 2]],
 labels=[[0, 0, 1, 1], [0, 1, 0, 1]])
 ```
在创建 Series 或 DataFrame 时，可以将这些对象作为 index 参数，或者通过 reindex 方法
更新 Series 或 DataFrame 的索引。

#### 2. 多级索引的等级名称
给 MultiIndex 的等级加上名称会为一些操作提供便利。你可以在前面任何一个 MultiIndex
构造器中通过 names 参数设置等级名称，也可以在创建之后通过索引的 names 属性来修改
名称：
```python
In[18]: pop.index.names = ['state', 'year']
 pop
Out[18]: state year
 California 2000 33871648
 2010 37253956
 New York 2000 18976457
 2010 19378102
 Texas 2000 20851820
 2010 25145561
 dtype: int64
 ```
在处理复杂的数据时，为等级设置名称是管理多个索引值的好办法。
#### 3. 多级列索引
每个 DataFrame 的行与列都是对称的，也就是说既然有多级行索引，那么同样可以有多级
列索引。让我们通过一份医学报告的模拟数据来演示：
```python
In[19]:
# 多级行列索引
index = pd.MultiIndex.from_product([[2013, 2014], [1, 2]],
 names=['year', 'visit'])
columns = pd.MultiIndex.from_product([['Bob', 'Guido', 'Sue'], ['HR', 'Temp']],
 names=['subject', 'type'])
# 模拟数据
data = np.round(np.random.randn(4, 6), 1)
data[:, ::2] *= 10
data += 37
# 创建DataFrame
health_data = pd.DataFrame(data, index=index, columns=columns)
health_data
Out[19]: subject Bob Guido Sue
 type HR Temp HR Temp HR Temp
 year visit
 2013 1 31.0 38.7 32.0 36.7 35.0 37.2
 2 44.0 37.7 50.0 35.0 29.0 36.7
 2014 1 30.0 37.4 39.0 37.8 61.0 36.9
 2 47.0 37.8 48.0 37.3 51.0 36.5
 ```
多级行列索引的创建非常简单。上面创建了一个简易的四维数据，四个维度分别为被检查
人的姓名、检查项目、检查年份和检查次数。可以在列索引的第一级查询姓名，从而获取
包含一个人（例如 Guido）全部检查信息的 DataFrame：
```python
In[20]: health_data['Guido']
Out[20]: type HR Temp 

 year visit
 2013 1 32.0 36.7
 2 50.0 35.0
 2014 1 39.0 37.8
 2 48.0 37.3
 ```
如果想获取包含多种标签的数据，需要通过对多个维度（姓名、国家、城市等标签）的多
次查询才能实现，这时使用多级行列索引进行查询会非常方便。
### 　多级索引的取值与切片
对 MultiIndex 的取值和切片操作很直观，你可以直接把索引看成额外增加的维度。我们先
来介绍 Series 多级索引的取值与切片方法，再介绍 DataFrame 的用法。
#### 1. Series多级索引
看看下面由各州历年人口数量创建的多级索引 Series：
```python
In[21]: pop
Out[21]: state year
 California 2000 33871648
 2010 37253956
 New York 2000 18976457
 2010 19378102
 Texas 2000 20851820
 2010 25145561
 dtype: int64
 ```
可以通过对多个级别索引值获取单个元素：
```python
In[22]: pop['California', 2000]
Out[22]: 33871648
```
MultiIndex 也支持局部取值（partial indexing），即只取索引的某一个层级。假如只取最高
级的索引，获得的结果是一个新的 Series，未被选中的低层索引值会被保留：
```python
In[23]: pop['California']
Out[23]: year
 2000 33871648
 2010 37253956
 dtype: int64
 ```
类似的还有局部切片，不过要求 MultiIndex 是按顺序排列的（就像将在 3.6.4 节介绍的
那样）：
```python
In[24]: pop.loc['California':'New York']
Out[24]: state year
 California 2000 33871648
 2010 37253956
 New York 2000 18976457
 2010 19378102
 dtype: int64
```
如果索引已经排序，那么可以用较低层级的索引取值，第一层级的索引可以用空切片：
```python
In[25]: pop[:, 2000]
Out[25]: state
 California 33871648
 New York 18976457
 Texas 20851820
 dtype: int64
 ```
其他取值与数据选择的方法（详情请参见 3.3 节）也都起作用。下面的例子是通过布尔掩
码选择数据：
```python
In[26]: pop[pop > 22000000]
Out[26]: state year
 California 2000 33871648
 2010 37253956
 Texas 2010 25145561
 dtype: int64
 ```
也可以用花哨的索引选择数据：
```python
In[27]: pop[['California', 'Texas']]
Out[27]: state year
 California 2000 33871648
 2010 37253956
 Texas 2000 20851820
 2010 25145561
 dtype: int64
 ```
#### 2. DataFrame多级索引
DataFrame 多级索引的用法与 Series 类似。还用之前的体检报告数据来演示：
```python
In[28]: health_data
Out[28]: subject Bob Guido Sue
 type HR Temp HR Temp HR Temp
 year visit
 2013 1 31.0 38.7 32.0 36.7 35.0 37.2
 2 44.0 37.7 50.0 35.0 29.0 36.7
 2014 1 30.0 37.4 39.0 37.8 61.0 36.9
 2 47.0 37.8 48.0 37.3 51.0 36.5
 ```
由于 DataFrame 的基本索引是列索引，因此 Series 中多级索引的用法到了 DataFrame 中就
应用在列上了。例如，可以通过简单的操作获取 Guido 的心率数据：
```python
In[29]: health_data['Guido', 'HR']
Out[29]: year visit
 2013 1 32.0
 2 50.0
 2014 1 39.0
 2 48.0
 Name: (Guido, HR), dtype: float64
```
与单索引类似，在 3.3 节介绍的 loc、iloc 和 ix 索引器都可以使用，例如：
```python
In[30]: health_data.iloc[:2, :2]
Out[30]: subject Bob
 type HR Temp
 year visit
 2013 1 31.0 38.7
 2 44.0 37.7
 ```
虽然这些索引器将多维数据当作二维数据处理，但是在 loc 和 iloc 中可以传递多个层级的
索引元组，例如：
```python
In[31]: health_data.loc[:, ('Bob', 'HR')]
Out[31]: year visit
 2013 1 31.0
 2 44.0
 2014 1 30.0
 2 47.0
 Name: (Bob, HR), dtype: float64
 ```
这种索引元组的用法不是很方便，如果在元组中使用切片还会导致语法错误：
```python
In[32]: health_data.loc[(:, 1), (:, 'HR')]
 File "<ipython-input-32-8e3cc151e316>", line 1
 health_data.loc[(:, 1), (:, 'HR')]
 ^
SyntaxError: invalid syntax
```
虽然你可以用 Python 内置的 slice() 函数获取想要的切片，但是还有一种更好的办法，就
是使用 IndexSlice 对象。Pandas 专门用它解决这类问题，例如：
```python
In[33]: idx = pd.IndexSlice
 health_data.loc[idx[:, 1], idx[:, 'HR']]
Out[33]: subject Bob Guido Sue
 type HR HR HR
 year visit
 2013 1 31.0 32.0 35.0
 2014 1 30.0 39.0 61.0
 ```
和带多级索引的 Series 和 DataFrame 进行数据交互的方法有很多，但就像本书中的诸多工
具一样，若想掌握它们，最好的办法就是使用它们！
###　多级索引行列转换
使用多级索引的关键是掌握有效数据转换的方法。Pandas 提供了许多操作，可以让数
据在内容保持不变的同时，按照需要进行行列转换。之前我们用一个简短的例子演示过
stack() 和 unstack() 的用法，但其实还有许多合理控制层级行列索引的方法，让我们来一
探究竟。

#### 1. 有序的索引和无序的索引
在前面的内容里，我们曾经简单提过多级索引排序，这里需要详细介绍一下。如果
MultiIndex 不是有序的索引，那么大多数切片操作都会失败。让我们演示一下。
首先创建一个不按字典顺序（lexographically）排列的多级索引 Series：
```python
In[34]: index = pd.MultiIndex.from_product([['a', 'c', 'b'], [1, 2]])
 data = pd.Series(np.random.rand(6), index=index)
 data.index.names = ['char', 'int']
 data
Out[34]: char int
 a 1 0.003001
 2 0.164974
 c 1 0.741650
 2 0.569264
 b 1 0.001693
 2 0.526226
 dtype: float64
 ```
如果想对索引使用局部切片，那么错误就会出现：
```python
In[35]: try:
 data['a':'b']
 except KeyError as e:
 print(type(e))
 print(e)
<class 'KeyError'>
'Key length (1) was greater than MultiIndex lexsort depth (0)'
```
尽管从错误信息里面看不出具体的细节，但问题是出在 MultiIndex 无序排列上。局部切片
和许多其他相似的操作都要求 MultiIndex 的各级索引是有序的（即按照字典顺序由 A 至
Z）。为此，Pandas 提供了许多便捷的操作完成排序，如 sort_index() 和 sortlevel() 方
法。我们用最简单的 sort_index() 方法来演示：
```python
In[36]: data = data.sort_index()
 data
Out[36]: char int
 a 1 0.003001
 2 0.164974
 b 1 0.001693
 2 0.526226
 c 1 0.741650
 2 0.569264
 dtype: float64
 ```
索引排序之后，局部切片就可以正常使用了：
```python
In[37]: data['a':'b']
Out[37]: char int
 a 1 0.003001 

 2 0.164974
 b 1 0.001693
 2 0.526226
 dtype: float64
 ```
#### 2. 索引stack与unstack
前文曾提过，我们可以将一个多级索引数据集转换成简单的二维形式，可以通过 level 参
数设置转换的索引层级：
```python
In[38]: pop.unstack(level=0)
Out[38]: state California New York Texas
 year
 2000 33871648 18976457 20851820
 2010 37253956 19378102 25145561
In[39]: pop.unstack(level=1)
Out[39]: year 2000 2010
 state
 California 33871648 37253956
 New York 18976457 19378102
 Texas 20851820 25145561
 ```
unstack() 是 stack() 的逆操作，同时使用这两种方法让数据保持不变：
```python
In[40]: pop.unstack().stack()
Out[40]: state year
 California 2000 33871648
 2010 37253956
 New York 2000 18976457
 2010 19378102
 Texas 2000 20851820
 2010 25145561
 dtype: int64
 ```
#### 3. 索引的设置与重置
层级数据维度转换的另一种方法是行列标签转换，可以通过 reset_index 方法实现。如
果在上面的人口数据 Series 中使用该方法，则会生成一个列标签中包含之前行索引标签
state 和 year 的 DataFrame。也可以用数据的 name 属性为列设置名称：
```python
In[41]: pop_flat = pop.reset_index(name='population')
 pop_flat
Out[41]: state year population
 0 California 2000 33871648
 1 California 2010 37253956
 2 New York 2000 18976457
 3 New York 2010 19378102
 4 Texas 2000 20851820
 5 Texas 2010 25145561
 ```
在解决实际问题的时候，如果能将类似这样的原始输入数据的列直接转换成 MultiIndex，

通常将大有裨益。其实可以通过 DataFrame 的 set_index 方法实现，返回结果就会是一个
带多级索引的 DataFrame：
```python
In[42]: pop_flat.set_index(['state', 'year'])
Out[42]: population
 state year
 California 2000 33871648
 2010 37253956
 New York 2000 18976457
 2010 19378102
 Texas 2000 20851820
 2010 25145561
 ```
在实践中，我发现用这种重建索引的方法处理数据集非常好用。
###　多级索引的数据累计方法
前面我们已经介绍过一些 Pandas 自带的数据累计方法，比如 mean()、sum() 和 max()。而
对于层级索引数据，可以设置参数 level 实现对数据子集的累计操作。
再一次以体检数据为例：
```python
In[43]: health_data
Out[43]: subject Bob Guido Sue
 type HR Temp HR Temp HR Temp
 year visit
 2013 1 31.0 38.7 32.0 36.7 35.0 37.2
 2 44.0 37.7 50.0 35.0 29.0 36.7
 2014 1 30.0 37.4 39.0 37.8 61.0 36.9
 2 47.0 37.8 48.0 37.3 51.0 36.5
 ```
如果你需要计算每一年各项指标的平均值，那么可以将参数 level 设置为索引 year：
```python
In[44]: data_mean = health_data.mean(level='year')
 data_mean
Out[44]: subject Bob Guido Sue
 type HR Temp HR Temp HR Temp
 year
 2013 37.5 38.2 41.0 35.85 32.0 36.95
 2014 38.5 37.6 43.5 37.55 56.0 36.70
 ```
如果再设置 axis 参数，就可以对列索引进行类似的累计操作了：
```python
In[45]: data_mean.mean(axis=1, level='type')
Out[45]: type HR Temp
 year
 2013 36.833333 37.000000
 2014 46.000000 37.283333
 ```
通过这两行数据，我们就可以获取每一年所有人的平均心率和体温了。这种语法其实就是

GroupBy 功能的快捷方式，我们将在 3.9 节详细介绍。尽管这只是一个简单的示例，但是其
原理和实际工作中遇到的情况类似。
#### Panel 数据
这里还有一些 Pandas 的基本数据结构没有介绍到，包括 pd.Panel 对象和 pd.Panel4D
对象。这两种数据结构可以分别看成是（一维数组）Series 和（二维数组）DataFrame
的三维与四维形式。如果你熟悉 Series 和 DataFrame 的使用方法，那么 Panel 和
Panel4D 使用起来也会很简单，ix、loc 和 iloc 索引器（详情请参见 3.3 节）在高维数
据结构上的用法更是完全相同。
但是本书并不打算进一步介绍这两种数据结构，我个人认为多级索引在大多数情况下
都是更实用、更直观的高维数据形式。另外，Panel 采用密集数据存储形式，而多级
索引采用稀疏数据存储形式。在解决许多真实的数据集时，随着维度的不断增加，密
集数据存储形式的效率将越来越低。但是这类数据结构对一些有特殊需求的应用还是
有用的。如果你想对 Panel 与 Panel4D 数据结构有更多的认识，请参见 3.14 节。
