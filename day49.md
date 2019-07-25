## 合并数据集：Concat与Append操作
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
 ```
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
```
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
```python
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

##　合并数据集：合并与连接

###　数据连接的类型
pd.merge() 函数实现了三种数据连接的类型：一对一、多对一和多对多。这三种数据连接
类型都通过 pd.merge() 接口进行调用，根据不同的数据连接需求进行不同的操作。下面将
通过一些示例来演示这三种类型，并进一步介绍更多的细节。
#### 1. 一对一连接
一对一连接可能是最简单的数据合并类型了，与 3.7 节介绍的按列合并十分相似。如下面
示例所示，有两个包含同一所公司员工不同信息的 DataFrame：
```python
In[2]:
df1 = pd.DataFrame({'employee': ['Bob', 'Jake', 'Lisa', 'Sue'],
 'group': ['Accounting', 'Engineering', 'Engineering', 'HR']})
df2 = pd.DataFrame({'employee': ['Lisa', 'Bob', 'Jake', 'Sue'],
 'hire_date': [2004, 2008, 2012, 2014]})
print(df1); print(df2)
df1 df2
 employee group employee hire_date
0 Bob Accounting 0 Lisa 2004
1 Jake Engineering 1 Bob 2008
2 Lisa Engineering 2 Jake 2012
3 Sue HR 3 Sue 2014
```
若想将这两个 DataFrame 合并成一个 DataFrame，可以用 pd.merge() 函数实现：
```python
In[3]: df3 = pd.merge(df1, df2)
 df3
Out[3]: employee group hire_date
 0 Bob Accounting 2008
 1 Jake Engineering 2012
 2 Lisa Engineering 2004
 3 Sue HR 2014
 ```
pd.merge() 方法会发现两个 DataFrame 都有“employee”列，并会自动以这列作为键进行
连接。两个输入的合并结果是一个新的 DataFrame。需要注意的是，共同列的位置可以是
不一致的。例如在这个例子中，虽然 df1 与 df2 中“employee”列的位置是不一样的，但
是 pd.merge() 函数会正确处理这个问题。另外还需要注意的是，pd.merge() 会默认丢弃原
来的行索引，不过也可以自定义（详情请参见 3.8.3 节）。
#### 2. 多对一连接
多对一连接是指，在需要连接的两个列中，有一列的值有重复。通过多对一连接获得的结
果 DataFrame 将会保留重复值。请看下面的例子：
```python
In[4]: df4 = pd.DataFrame({'group': ['Accounting', 'Engineering', 'HR'],
 'supervisor': ['Carly', 'Guido', 'Steve']})
 print(df3); print(df4); print(pd.merge(df3, df4))
df3 df4
 employee group hire_date group supervisor
0 Bob Accounting 2008 0 Accounting Carly
1 Jake Engineering 2012 1 Engineering Guido
2 Lisa Engineering 2004 2 HR Steve
3 Sue HR 2014
pd.merge(df3, df4)
 employee group hire_date supervisor
0 Bob Accounting 2008 Carly
1 Jake Engineering 2012 Guido
2 Lisa Engineering 2004 Guido
3 Sue HR 2014 Steve
```
在结果 DataFrame 中多了一个“supervisor”列，里面有些值会因为输入数据的对应关系而
有所重复。
#### 3. 多对多连接
多对多连接是个有点儿复杂的概念，不过也可以理解。如果左右两个输入的共同列都包含
重复值，那么合并的结果就是一种多对多连接。用一个例子来演示可能更容易理解。来看
下面的例子，里面有一个 DataFrame 显示不同岗位人员的一种或多种能力。
通过多对多链接，就可以得知每位员工所具备的能力：
```python
In[5]: df5 = pd.DataFrame({'group': ['Accounting', 'Accounting',
 'Engineering', 'Engineering', 'HR', 'HR'],
 'skills': ['math', 'spreadsheets', 'coding', 'linux',
 'spreadsheets', 'organization']})
print(df1); print(df5); print(pd.merge(df1, df5))
df1 df5
 employee group group skills
0 Bob Accounting 0 Accounting math
1 Jake Engineering 1 Accounting spreadsheets
2 Lisa Engineering 2 Engineering coding
3 Sue HR 3 Engineering linux
 4 HR spreadsheets
 5 HR organization
pd.merge(df1, df5)
 employee group skills
0 Bob Accounting math
1 Bob Accounting spreadsheets
2 Jake Engineering coding
3 Jake Engineering linux
4 Lisa Engineering coding
5 Lisa Engineering linux
6 Sue HR spreadsheets
7 Sue HR organization
```
这三种数据连接类型可以直接与其他 Pandas 工具组合使用，从而实现各种各样的功
能。但是工作中的真实数据集往往并不像示例中演示的那么干净、整洁。下面就来介绍
pd.merge() 的一些功能，它们可以让你更好地应对数据连接中的问题。
###　设置数据合并的键
我们已经见过 pd.merge() 的默认行为：它会将两个输入的一个或多个共同列作为键进行合
并。但由于两个输入要合并的列通常都不是同名的，因此 pd.merge() 提供了一些参数处理
这个问题。
#### 1. 参数on的用法
最简单的方法就是直接将参数 on 设置为一个列名字符串或者一个包含多列名称的列表：
```python
In[6]: print(df1); print(df2); print(pd.merge(df1, df2, on='employee'))
df1 df2
 employee group employee hire_date
0 Bob Accounting 0 Lisa 2004
1 Jake Engineering 1 Bob 2008
2 Lisa Engineering 2 Jake 2012
3 Sue HR 3 Sue 2014
pd.merge(df1, df2, on='employee')
 employee group hire_date
0 Bob Accounting 2008
1 Jake Engineering 2012
2 Lisa Engineering 2004
3 Sue HR 2014
```
这个参数只能在两个 DataFrame 有共同列名的时候才可以使用。
#### 2. left_on与right_on参数
有时你也需要合并两个列名不同的数据集，例如前面的员工信息表中有一个字段不是
“employee”而是“name”。在这种情况下，就可以用 left_on 和 right_on 参数来指定
列名：
```python
In[7]:
df3 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
 'salary': [70000, 80000, 120000, 90000]})
print(df1); print(df3);
print(pd.merge(df1, df3, left_on="employee", right_on="name"))
df1 df3
 employee group name salary
0 Bob Accounting 0 Bob 70000
1 Jake Engineering 1 Jake 80000
2 Lisa Engineering 2 Lisa 120000
3 Sue HR 3 Sue 90000
pd.merge(df1, df3, left_on="employee", right_on="name")

 employee group name salary
0 Bob Accounting Bob 70000
1 Jake Engineering Jake 80000
2 Lisa Engineering Lisa 120000
3 Sue HR Sue 90000
```
获取的结果中会有一个多余的列，可以通过 DataFrame 的 drop() 方法将这列去掉：
```python
In[8]:
pd.merge(df1, df3, left_on="employee", right_on="name").drop('name', axis=1)
Out[8]: employee group salary
 0 Bob Accounting 70000
 1 Jake Engineering 80000
 2 Lisa Engineering 120000
 3 Sue HR 90000
 ```
#### 3. left_index与right_index参数
除了合并列之外，你可能还需要合并索引。就像下面例子中的数据那样：
```python
In[9]: df1a = df1.set_index('employee')
 df2a = df2.set_index('employee')
 print(df1a); print(df2a)
df1a df2a
 group hire_date
employee employee
Bob Accounting Lisa 2004
Jake Engineering Bob 2008
Lisa Engineering Jake 2012
Sue HR Sue 2014
```
你可以通过设置 pd.merge() 中的 left_index 和 / 或 right_index 参数将索引设置为键来实
现合并：
```python
In[10]:
print(df1a); print(df2a);
print(pd.merge(df1a, df2a, left_index=True, right_index=True))
df1a df2a
 group hire_date
employee employee
Bob Accounting Lisa 2004
Jake Engineering Bob 2008
Lisa Engineering Jake 2012
Sue HR Sue 2014
pd.merge(df1a, df2a, left_index=True, right_index=True)
 group hire_date
employee
Lisa Engineering 2004
Bob Accounting 2008
Jake Engineering 2012
Sue HR 2014
```
为了方便考虑，DataFrame 实现了 join() 方法，它可以按照索引进行数据合并：
```python
In[11]: print(df1a); print(df2a); print(df1a.join(df2a))
df1a df2a
 group hire_date
employee employee
Bob Accounting Lisa 2004
Jake Engineering Bob 2008
Lisa Engineering Jake 2012
Sue HR Sue 2014
df1a.join(df2a)
 group hire_date
employee
Bob Accounting 2008
Jake Engineering 2012
Lisa Engineering 2004
Sue HR 2014
```
如果想将索引与列混合使用，那么可以通过结合 left_index 与 right_on，或者结合 left_
on 与 right_index 来实现：
```python
In[12]:
print(df1a); print(df3);
print(pd.merge(df1a, df3, left_index=True, right_on='name'))
df1a df3
 group
employee name salary
Bob Accounting 0 Bob 70000
Jake Engineering 1 Jake 80000
Lisa Engineering 2 Lisa 120000
Sue HR 3 Sue 90000
pd.merge(df1a, df3, left_index=True, right_on='name')
 group name salary
0 Accounting Bob 70000
1 Engineering Jake 80000
2 Engineering Lisa 120000
3 HR Sue 90000
```
当然，这些参数都适用于多个索引和 / 或多个列名，函数接口非常简单。若想了解 Pandas
数据合并的更多信息，请参考 Pandas 文档中“Merge, Join, and Concatenate”（http://pandas.
pydata.org/pandas-docs/stable/merging.html）节。
### 　设置数据连接的集合操作规则
通过前面的示例，我们总结出数据连接的一个重要条件：集合操作规则。当一个值出现在
一列，却没有出现在另一列时，就需要考虑集合操作规则了。来看看下面的例子：
```python
In[13]: df6 = pd.DataFrame({'name': ['Peter', 'Paul', 'Mary'],
 'food': ['fish', 'beans', 'bread']},
 columns=['name', 'food'])
 df7 = pd.DataFrame({'name': ['Mary', 'Joseph'],
 'drink': ['wine', 'beer']},
 columns=['name', 'drink'])
 print(df6); print(df7); print(pd.merge(df6, df7))
df6 df7 pd.merge(df6, df7)
 name food name drink name food drink
0 Peter fish 0 Mary wine 0 Mary bread wine
1 Paul beans 1 Joseph beer
2 Mary bread
```
我们合并两个数据集，在“name”列中只有一个共同的值：Mary。默认情况下，结果中只
会包含两个输入集合的交集，这种连接方式被称为内连接（inner join）。我们可以用 how 参
数设置连接方式，默认值为 'inner'：
```python
In[14]: pd.merge(df6, df7, how='inner')
Out[14]: name food drink
 0 Mary bread wine
 ```
how 参数支持的数据连接方式还有 'outer'、'left' 和 'right'。外连接（outer join）返回
两个输入列的交集，所有缺失值都用 NaN 填充：
```python
In[15]: print(df6); print(df7); print(pd.merge(df6, df7, how='outer'))
df6 df7 pd.merge(df6, df7, how='outer')
 name food name drink name food drink
0 Peter fish 0 Mary wine 0 Peter fish NaN
1 Paul beans 1 Joseph beer 1 Paul beans NaN
2 Mary bread 2 Mary bread wine
 3 Joseph NaN beer
 ```
左连接（left join）和右连接（right join）返回的结果分别只包含左列和右列，如下所示：
```python
In[16]: print(df6); print(df7); print(pd.merge(df6, df7, how='left'))
df6 df7 pd.merge(df6, df7, how='left')
 name food name drink name food drink
0 Peter fish 0 Mary wine 0 Peter fish NaN
1 Paul beans 1 Joseph beer 1 Paul beans NaN
2 Mary bread 2 Mary bread wine
```
现在输出的行中只包含左边输入列的值。如果用 how='right' 的话，输出的行则只包含右
边输入列的值。
这四种数据连接的集合操作规则都可以直接应用于前面介绍过的连接类型。
### 重复列名：suffixes参数
最后，你可能会遇到两个输入 DataFrame 有重名列的情况。来看看下面的例子：
```python
In[17]: df8 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
 'rank': [1, 2, 3, 4]})
 df9 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
 'rank': [3, 1, 4, 2]})
 print(df8); print(df9); print(pd.merge(df8, df9, on="name"))
df8 df9 pd.merge(df8, df9, on="name")
 name rank name rank name rank_x rank_y
0 Bob 1 0 Bob 3 0 Bob 1 3
1 Jake 2 1 Jake 1 1 Jake 2 1
2 Lisa 3 2 Lisa 4 2 Lisa 3 4
3 Sue 4 3 Sue 2 3 Sue 4 2
```
由于输出结果中有两个重复的列名，因此 pd.merge() 函数会自动为它们增加后缀 _x _或 _y_* ，
当然也可以通过 suffixes 参数自定义后缀名：
```python
In[18]:
print(df8); print(df9);
print(pd.merge(df8, df9, on="name", suffixes=["_L", "_R"]))
df8 df9
 name rank name rank
0 Bob 1 0 Bob 3
1 Jake 2 1 Jake 1
2 Lisa 3 2 Lisa 4
3 Sue 4 3 Sue 2
pd.merge(df8, df9, on="name", suffixes=["_L", "_R"])
 name rank_L rank_R
0 Bob 1 3
1 Jake 2 1
2 Lisa 3 4
3 Sue 4 2
```
suffixes 参数同样适用于任何连接方式，即使有三个及三个以上的重复列名时也同样适用。
关于关系代数的更多信息，请参见 3.9 节，里面对关系代数进行了更加深入的介绍。另外，
还可以参考 Pandas 文档中“Merge, Join, and Concatenate”（http://pandas.pydata.org/pandasdocs/stable/merging.html）节。
### 案例：美国各州的统计数据
数据的合并与连接是组合来源不同的数据的最常用方法。下面通过美国各州的统计数据来
进行一个演示，请到 https://github.com/jakevdp/data-USstates/ 下载数据：
```python
In[19]:
# 请使用下面的shell下载数据
# !curl -O https://raw.githubusercontent.com/jakevdp/
# data-USstates/master/state-population.csv
# !curl -O https://raw.githubusercontent.com/jakevdp/
# data-USstates/master/state-areas.csv
# !curl -O https://raw.githubusercontent.com/jakevdp/
# data-USstates/master/state-abbrevs.csv
```
用 Pandas 的 read_csv() 函数看看这三个数据集：
```python
In[20]: pop = pd.read_csv('state-population.csv')
 areas = pd.read_csv('state-areas.csv')
 abbrevs = pd.read_csv('state-abbrevs.csv')
 print(pop.head()); print(areas.head()); print(abbrevs.head())
pop.head() areas.head()
 state/region ages year population state area (sq. mi)
0 AL under18 2012 1117489.0 0 Alabama 52423
1 AL total 2012 4817528.0 1 Alaska 656425
2 AL under18 2010 1130966.0 2 Arizona 114006
3 AL total 2010 4785570.0 3 Arkansas 53182
4 AL under18 2011 1125763.0 3 Arkansas 53182
 4 California 163707
abbrevs.head()
 state abbreviation
0 Alabama AL
1 Alaska AK
2 Arizona AZ
3 Arkansas AR
4 California CA
```
看过这些数据之后，我们想要计算一个比较简单的指标：美国各州的人口密度排名。虽然
可以直接通过计算每张表获取结果，但这次试着用数据集连接来解决这个问题。
首先用一个多对一合并获取人口（pop）DataFrame 中各州名称缩写对应的全称。我们需要
将 pop 的 state/region 列与 abbrevs 的 abbreviation 列进行合并，还需要通过 how='outer'
确保数据没有丢失。
```python
In[21]: merged = pd.merge(pop, abbrevs, how='outer',
 left_on='state/region', right_on='abbreviation')
 merged = merged.drop('abbreviation', 1) # 丢弃重复信息
 merged.head()
Out[21]: state/region ages year population state
 0 AL under18 2012 1117489.0 Alabama
 1 AL total 2012 4817528.0 Alabama
 2 AL under18 2010 1130966.0 Alabama
 3 AL total 2010 4785570.0 Alabama
 4 AL under18 2011 1125763.0 Alabama
 ```
来全面检查一下数据是否有缺失，我们可以对每个字段逐行检查是否有缺失值：
```python
In[22]: merged.isnull().any()
Out[22]: state/region False
 ages False
 year False
 population True
 state True
 dtype: bool
```
部分 population 是缺失值，让我们仔细看看那些数据！
```python
In[23]: merged[merged['population'].isnull()].head()
Out[23]: state/region ages year population state
 2448 PR under18 1990 NaN NaN
 2449 PR total 1990 NaN NaN
 2450 PR total 1991 NaN NaN
 2451 PR under18 1991 NaN NaN
 2452 PR total 1993 NaN NaN
 ```
好像所有的人口缺失值都出现在 2000 年之前的波多黎各 2
，此前并没有统计过波多黎各的人口。
更重要的是，我们还发现一些新的州的数据也有缺失，可能是由于名称缩写没有匹配上全
程！来看看究竟是哪个州有缺失：
```python
In[24]: merged.loc[merged['state'].isnull(), 'state/region'].unique()
Out[24]: array(['PR', 'USA'], dtype=object)
```
我们可以快速解决这个问题：人口数据中包含波多黎各（PR）和全国总数（USA），但这
两项没有出现在州名称缩写表中。来快速填充对应的全称：
```python
In[25]: merged.loc[merged['state/region'] == 'PR', 'state'] = 'Puerto Rico'
 merged.loc[merged['state/region'] == 'USA', 'state'] = 'United States'
 merged.isnull().any()
Out[25]: state/region False
 ages False
 year False
 population True
 state False
 dtype: bool
 ```
现在 state 列没有缺失值了，万事俱备！
让我们用类似的规则将面积数据也合并进来。用两个数据集共同的 state 列来合并：
```python
In[26]: final = pd.merge(merged, areas, on='state', how='left')
 final.head()
Out[26]: state/region ages year population state area (sq. mi)
 0 AL under18 2012 1117489.0 Alabama 52423.0
 1 AL total 2012 4817528.0 Alabama 52423.0
 2 AL under18 2010 1130966.0 Alabama 52423.0
 3 AL total 2010 4785570.0 Alabama 52423.0
 4 AL under18 2011 1125763.0 Alabama 52423.0
 ```
再检查一下数据，看看哪些列还有缺失值，没有匹配上：
```python
In[27]: final.isnull().any()
Out[27]: state/region False
 ages False


 year False
 population True
 state False
 area (sq. mi) True
 dtype: bool
  ```
注 2： Puerto Rico，目前尚未成为美国的第 51 个州，2017 年 6 月第五次入美公投。——译者注
面积 area 列里面还有缺失值。来看看究竟是哪些地区面积缺失：
 ```
注 2： Puerto Rico，目前尚未成为美国的第 51 个州，2017 年 6 月第五次入美公投。——译者注
In[28]: final['state'][final['area (sq. mi)'].isnull()].unique()
Out[28]: array(['United States'], dtype=object)
```
我们发现面积（areas）DataFrame 里面不包含全美国的面积数据。可以插入全国总面积数
据（对各州面积求和即可），但是针对本案例，我们要去掉这个缺失值，因为全国的人口
密度在此无关紧要：
 ```
注 2： Puerto Rico，目前尚未成为美国的第 51 个州，2017 年 6 月第五次入美公投。——译者注
In[29]: final.dropna(inplace=True)
 final.head()
Out[29]: state/region ages year population state area (sq. mi)
 0 AL under18 2012 1117489.0 Alabama 52423.0
 1 AL total 2012 4817528.0 Alabama 52423.0
 2 AL under18 2010 1130966.0 Alabama 52423.0
 3 AL total 2010 4785570.0 Alabama 52423.0
 4 AL under18 2011 1125763.0 Alabama 52423.0
 ```
现在所有的数据都准备好了。为了解决眼前的问题，先选择 2000 年的各州人口以及总人
口数据。让我们用 query() 函数进行快速计算（这需要用到 numexpr 程序库，详情请参见
3.13 节）：
```python
In[30]: data2010 = final.query("year == 2010 & ages == 'total'")
 data2010.head()
Out[30]: state/region ages year population state area (sq. mi)
 3 AL total 2010 4785570.0 Alabama 52423.0
 91 AK total 2010 713868.0 Alaska 656425.0
 101 AZ total 2010 6408790.0 Arizona 114006.0
 189 AR total 2010 2922280.0 Arkansas 53182.0
 197 CA total 2010 37333601.0 California 163707.0
 ```
现在来计算人口密度并按序排列。首先对索引进行重置，然后再计算结果：
```python
In[31]: data2010.set_index('state', inplace=True)
 density = data2010['population'] / data2010['area (sq. mi)']
In[32]: density.sort_values(ascending=False, inplace=True)
 density.head()
Out[32]: state
 District of Columbia 8898.897059
 Puerto Rico 1058.665149
 New Jersey 1009.253268
 Rhode Island 681.339159
 Connecticut 645.600649
 dtype: float64
```
计算结果是美国各州加上华盛顿特区（Washington, DC）、波多黎各在 2010 年的人口密度
排序，以万人 / 平方英里为单位。我们发现人口密度最高的地区是华盛顿特区的哥伦比亚
地区（the District of Columbia）。在各州的人口密度中，新泽西州（New Jersey）是最高的。
还可以看看人口密度最低的几个州的数据：
```python
In[33]: density.tail()
Out[33]: state
 South Dakota 10.583512
 North Dakota 9.537565
 Montana 6.736171
 Wyoming 5.768079
 Alaska 1.087509
 dtype: float64
 ```
可以看出，人口密度最低的州是阿拉斯加（Alaska），刚刚超过 1 万人 / 平方英里。
当人们用现实世界的数据解决问题时，合并这类脏乱的数据是十分常见的任务。希望这个
案例可以帮你把前面介绍过的工具串起来，从而在数据中找到想要的答案！
## 累计与分组
在对较大的数据进行分析时，一项基本的工作就是有效的数据累计（summarization）：计
算累计（aggregation）指标，如 sum()、mean()、median()、min() 和 max()，其中每一个指
标都呈现了大数据集的特征。在这一节中，我们将探索 Pandas 的累计功能，从类似前面
NumPy 数组中的简单操作，到基于 groupby 实现的复杂操作。
### 行星数据
我们将通过 Seaborn 程序库（http://seaborn.pydata.org，详情请参见 4.16 节）用一份行星数
据来进行演示，其中包含天文学家观测到的围绕恒星运转的行星数据（通常简称为太阳系
外行星或外行星）。行星数据可以直接通过 Seaborn 下载：
```python
In[2]: import seaborn as sns
 planets = sns.load_dataset('planets')
 planets.shape
Out[2]: (1035, 6)
In[3]: planets.head()
Out[3]: method number orbital_period mass distance year
 0 Radial Velocity 1 269.300 7.10 77.40 2006
 1 Radial Velocity 1 874.774 2.21 56.95 2008
 2 Radial Velocity 1 763.000 2.60 19.84 2011
 3 Radial Velocity 1 326.030 19.40 110.62 2007
 4 Radial Velocity 1 516.220 10.50 119.47 2009
 ```
数据中包含了截至 2014 年已被发现的一千多颗外行星的资料。

### Pandas的简单累计功能
之前我们介绍过 NumPy 数组的一些数据累计指标（详情请参见 2.4 节）。与一维 NumPy 数
组相同，Pandas 的 Series 的累计函数也会返回一个统计值：
```python
In[4]: rng = np.random.RandomState(42)
 ser = pd.Series(rng.rand(5))
 ser
Out[4]: 0 0.374540
 1 0.950714
 2 0.731994
 3 0.598658
 4 0.156019
 dtype: float64
In[5]: ser.sum()
Out[5]: 2.8119254917081569
In[6]: ser.mean()
Out[6]: 0.56238509834163142
```
DataFrame 的累计函数默认对每列进行统计：
```python
In[7]: df = pd.DataFrame({'A': rng.rand(5),
 'B': rng.rand(5)})
 df
Out[7]: A B
 0 0.155995 0.020584
 1 0.058084 0.969910
 2 0.866176 0.832443
 3 0.601115 0.212339
 4 0.708073 0.181825
In[8]: df.mean()
Out[8]: A 0.477888
 B 0.443420
 dtype: float64
 ```
设置 axis 参数，你就可以对每一行进行统计了：
```python
In[9]: df.mean(axis='columns')
Out[9]: 0 0.088290
 1 0.513997
 2 0.849309
 3 0.406727
 4 0.444949
 dtype: float64
 ```
Pandas 的 Series 和 DataFrame 支持所有 2.4 节中介绍的常用累计函数。另外，还有一个非

常方便的 describe() 方法可以计算每一列的若干常用统计值。让我们在行星数据上试验一
下，首先丢弃有缺失值的行：
```python
In[10]: planets.dropna().describe()
Out[10]: number orbital_period mass distance year
 count 498.00000 498.000000 498.000000 498.000000 498.000000
 mean 1.73494 835.778671 2.509320 52.068213 2007.377510
 std 1.17572 1469.128259 3.636274 46.596041 4.167284
 min 1.00000 1.328300 0.003600 1.350000 1989.000000
 25% 1.00000 38.272250 0.212500 24.497500 2005.000000
 50% 1.00000 357.000000 1.245000 39.940000 2009.000000
 75% 2.00000 999.600000 2.867500 59.332500 2011.000000
 max 6.00000 17337.500000 25.000000 354.000000 2014.000000
 ```
这是一种理解数据集所有统计属性的有效方法。例如，从年份 year 列中可以看出，1989
年首次发现外行星，而且一半的已知外行星都是在 2010 年及以后的年份被发现的。这主
要得益于开普勒计划——一个通过激光望远镜发现恒星周围椭圆轨道行星的太空计划。
Pandas 内置的一些累计方法如表 3-3 所示。
表3-3：Pandas的累计方法
指标 描述
count() 计数项
first()、last() 第一项与最后一项
mean()、median() 均值与中位数
min()、max() 最小值与最大值
std()、var() 标准差与方差
mad() 均值绝对偏差（mean absolute deviation）
prod() 所有项乘积
sum() 所有项求和
DataFrame 和 Series 对象支持以上所有方法。
但若想深入理解数据，仅仅依靠累计函数是远远不够的。数据累计的下一级别是 groupby
操作，它可以让你快速、有效地计算数据各子集的累计值。
### GroupBy：分割、应用和组合
简单的累计方法可以让我们对数据集有一个笼统的认识，但是我们经常还需要对某些标签
或索引的局部进行累计分析，这时就需要用到 groupby 了。虽然“分组”（group by）这个
名字是借用 SQL 数据库语言的命令，但其理念引用发明 R 语言 frame 的 Hadley Wickham
的观点可能更合适：分割（split）、应用（apply）和组合（combine）。
#### 1. 分割、应用和组合
一个经典分割 - 应用 - 组合操作示例如图 3-1 所示，其中“apply”的是一个求和函数。

图 3-1 清晰地描述了 GroupBy 的过程。
• 分割步骤将 DataFrame 按照指定的键分割成若干组。
• 应用步骤对每个组应用函数，通常是累计、转换或过滤函数。
• 组合步骤将每一组的结果合并成一个输出数组。
虽然我们也可以通过前面介绍的一系列的掩码、累计与合并操作来实现，但是意识到中间
分割过程不需要显式地暴露出来这一点十分重要。而且 GroupBy（经常）只需要一行代码，
就可以计算每组的和、均值、计数、最小值以及其他累计值。GroupBy 的用处就是将这些
步骤进行抽象：用户不需要知道在底层如何计算，只要把操作看成一个整体就够了。
用 Pandas 进行图 3-1 所示的计算作为具体的示例。从创建输入 DataFrame 开始：
```python
In[11]: df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'],
 'data': range(6)}, columns=['key', 'data'])
 df
Out[11]: key data
 0 A 0
 1 B 1
 2 C 2
 3 A 3
 4 B 4
 5 C 5
 ```
我们可以用 DataFrame 的 groupby() 方法进行绝大多数常见的分割 - 应用 - 组合操作，将
需要分组的列名传进去即可：
```python
In[12]: df.groupby('key')
Out[12]: <pandas.core.groupby.DataFrameGroupBy object at 0x117272160>
```
需要注意的是，这里的返回值不是一个 DataFrame 对象，而是一个 DataFrameGroupBy 对象。
这个对象的魔力在于，你可以将它看成是一种特殊形式的 DataFrame，里面隐藏着若干组
数据，但是在没有应用累计函数之前不会计算。这种“延迟计算”（lazy evaluation）的方
法使得大多数常见的累计操作可以通过一种对用户而言几乎是透明的（感觉操作仿佛不存
在）方式非常高效地实现。
为了得到这个结果，可以对 DataFrameGroupBy 对象应用累计函数，它会完成相应的应用 /
组合步骤并生成结果：
```python
In[13]: df.groupby('key').sum()
Out[13]: data
 key
 A 3
 B 5
 C 7
 ```
sum() 只是众多可用方法中的一个。你可以用 Pandas 或 NumPy 的任意一种累计函数，也
可以用任意有效的 DataFrame 对象。下面就会介绍。
#### 2. GroupBy对象
GroupBy 对象是一种非常灵活的抽象类型。在大多数场景中，你可以将它看成是 DataFrame
的集合，在底层解决所有难题。让我们用行星数据来做一些演示。
GroupBy 中最重要的操作可能就是 aggregate、filter、transform 和 apply（累计、过滤、转
换、应用）了，后文将详细介绍这些内容，现在先来介绍一些 GroupBy 的基本操作方法。
(1) 按列取值。GroupBy 对象与 DataFrame 一样，也支持按列取值，并返回一个修改过的
GroupBy 对象，例如：
```python
In[14]: planets.groupby('method')
Out[14]: <pandas.core.groupby.DataFrameGroupBy object at 0x1172727b8>
In[15]: planets.groupby('method')['orbital_period']
Out[15]: <pandas.core.groupby.SeriesGroupBy object at 0x117272da0>
```
这里从原来的 DataFrame 中取某个列名作为一个 Series 组。与 GroupBy 对象一样，直到
我们运行累计函数，才会开始计算：
```python
In[16]: planets.groupby('method')['orbital_period'].median()
Out[16]: method
 Astrometry 631.180000
 Eclipse Timing Variations 4343.500000
 Imaging 27500.000000
 Microlensing 3300.000000
 Orbital Brightness Modulation 0.342887
 Pulsar Timing 66.541900
 Pulsation Timing Variations 1170.000000
 Radial Velocity 360.200000 

 Transit 5.714932
 Transit Timing Variations 57.011000
 Name: orbital_period, dtype: float64
 ```
这样就可以获得不同方法下所有行星公转周期（按天计算）的中位数。
(2) 按组迭代。GroupBy 对象支持直接按组进行迭代，返回的每一组都是 Series 或 DataFrame：
```python
In[17]: for (method, group) in planets.groupby('method'):
 print("{0:30s} shape={1}".format(method, group.shape))
Astrometry shape=(2, 6)
Eclipse Timing Variations shape=(9, 6)
Imaging shape=(38, 6)
Microlensing shape=(23, 6)
Orbital Brightness Modulation shape=(3, 6)
Pulsar Timing shape=(5, 6)
Pulsation Timing Variations shape=(1, 6)
Radial Velocity shape=(553, 6)
Transit shape=(397, 6)
Transit Timing Variations shape=(4, 6)
```
尽管通常还是使用内置的 apply 功能速度更快，但这种方式在手动处理某些问题时非常
有用，后面会详细介绍。
(3) 调用方法。借助 Python 类的魔力（@classmethod），可以让任何不由 GroupBy 对象直接
实现的方法直接应用到每一组，无论是 DataFrame 还是 Series 对象都同样适用。例如，
你可以用 DataFrame 的 describe() 方法进行累计，对每一组数据进行描述性统计：
```python
In[18]: planets.groupby('method')['year'].describe().unstack()
Out[18]:
 count mean std min 25% \\
method
Astrometry 2.0 2011.500000 2.121320 2010.0 2010.75
Eclipse Timing Variations 9.0 2010.000000 1.414214 2008.0 2009.00
Imaging 38.0 2009.131579 2.781901 2004.0 2008.00
Microlensing 23.0 2009.782609 2.859697 2004.0 2008.00
Orbital Brightness Modulation 3.0 2011.666667 1.154701 2011.0 2011.00
Pulsar Timing 5.0 1998.400000 8.384510 1992.0 1992.00
Pulsation Timing Variations 1.0 2007.000000 NaN 2007.0 2007.00
Radial Velocity 553.0 2007.518987 4.249052 1989.0 2005.00
Transit 397.0 2011.236776 2.077867 2002.0 2010.00
Transit Timing Variations 4.0 2012.500000 1.290994 2011.0 2011.75
 50% 75% max
method
Astrometry 2011.5 2012.25 2013.0
Eclipse Timing Variations 2010.0 2011.00 2012.0
Imaging 2009.0 2011.00 2013.0
Microlensing 2010.0 2012.00 2013.0
Orbital Brightness Modulation 2011.0 2012.00 2013.0
Pulsar Timing 1994.0 2003.00 2011.0
Pulsation Timing Variations 2007.0 2007.00 2007.0 

Radial Velocity 2009.0 2011.00 2014.0
Transit 2012.0 2013.00 2014.0
Transit Timing Variations 2012.5 2013.25 2014.0
```
这张表可以帮助我们对数据有更深刻的认识，例如大多数行星都是通过 Radial Velocity
和 Transit 方法发现的，而且后者在近十年变得越来越普遍（得益于更新、更精确的望远
镜）。最新的 Transit Timing Variation 和 Orbital Brightness Modulation 方法在 2011 年之
后才有新的发现。
这只是演示 Pandas 调用方法的示例之一。方法首先会应用到每组数据上，然后结果由
GroupBy 组合后返回。另外，任意 DataFrame / Series 的方法都可以由 GroupBy 方法调用，
从而实现非常灵活强大的操作。
#### 3. 累计、过滤、转换和应用
虽然前面的章节只重点介绍了组合操作，但是还有许多操作没有介绍，尤其是 GroupBy 对
象的 aggregate()、filter()、transform() 和 apply() 方法，在数据组合之前实现了大量
高效的操作。
为了方便后面内容的演示，使用下面这个 DataFrame：
```python
In[19]: rng = np.random.RandomState(0)
 df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'],
 'data1': range(6),
 'data2': rng.randint(0, 10, 6)},
 columns = ['key', 'data1', 'data2'])
 df
Out[19]: key data1 data2
 0 A 0 5
 1 B 1 0
 2 C 2 3
 3 A 3 3
 4 B 4 7
 5 C 5 9
 ```
(1) 累计。我们目前比较熟悉的 GroupBy 累计方法只有 sum() 和 median() 之类的简单函数，
但是 aggregate() 其实可以支持更复杂的操作，比如字符串、函数或者函数列表，并且
能一次性计算所有累计值。下面来快速演示一个例子：
```python
In[20]: df.groupby('key').aggregate(['min', np.median, max])
Out[20]: data1 data2
 min median max min median max
 key
 A 0 1.5 3 3 4.0 5
 B 1 2.5 4 0 3.5 7
 C 2 3.5 5 3 6.0 9
 ```
另一种用法就是通过 Python 字典指定不同列需要累计的函数：
```python
In[21]: df.groupby('key').aggregate({'data1': 'min',
 'data2': 'max'}) 
Out[21]: data1 data2
 key
 A 0 5
 B 1 7
 C 2 9
 ```
(2) 过滤。过滤操作可以让你按照分组的属性丢弃若干数据。例如，我们可能只需要保留标
准差超过某个阈值的组：
```python
In[22]:
def filter_func(x):
 return x['data2'].std() > 4
print(df); print(df.groupby('key').std());
print(df.groupby('key').filter(filter_func))
df df.groupby('key').std()
 key data1 data2 key data1 data2
0 A 0 5 A 2.12132 1.414214
1 B 1 0 B 2.12132 4.949747
2 C 2 3 C 2.12132 4.242641
3 A 3 3
4 B 4 7
5 C 5 9
df.groupby('key').filter(filter_func)
 key data1 data2
1 B 1 0
2 C 2 3
4 B 4 7
5 C 5 9
```
filter() 函数会返回一个布尔值，表示每个组是否通过过滤。由于 A 组 'data2' 列的
标准差不大于 4，所以被丢弃了。
(3) 转换。累计操作返回的是对组内全量数据缩减过的结果，而转换操作会返回一个新的全
量数据。数据经过转换之后，其形状与原来的输入数据是一样的。常见的例子就是将每
一组的样本数据减去各组的均值，实现数据标准化：
```python
In[23]: df.groupby('key').transform(lambda x: x - x.mean())
Out[23]: data1 data2
 0 -1.5 1.0
 1 -1.5 -3.5
 2 -1.5 -3.0
 3 1.5 -1.0
 4 1.5 3.5
 5 1.5 3.0
 ```
(4) apply() 方法。apply() 方法让你可以在每个组上应用任意方法。这个函数输入一个
DataFrame，返回一个 Pandas 对象（DataFrame 或 Series）或一个标量（scalar，单个数
值）。组合操作会适应返回结果类型。
下面的例子就是用 apply() 方法将第一列数据以第二列的和为基数进行标准化：
```python
In[24]: def norm_by_data2(x):
 # x是一个分组数据的DataFrame
 x['data1'] /= x['data2'].sum()
 return x
 print(df); print(df.groupby('key').apply(norm_by_data2))
df df.groupby('key').apply(norm_by_data2)
 key data1 data2 key data1 data2
0 A 0 5 0 A 0.000000 5
1 B 1 0 1 B 0.142857 0
2 C 2 3 2 C 0.166667 3
3 A 3 3 3 A 0.375000 3
4 B 4 7 4 B 0.571429 7
5 C 5 9 5 C 0.416667 9
```
GroupBy 里的 apply() 方法非常灵活，唯一需要注意的地方是它总是输入分组数据的
DataFrame，返回 Pandas 对象或标量。具体如何选择需要视情况而定。
#### 4. 设置分割的键
前面的简单例子一直在用列名分割 DataFrame。这只是众多分组操作中的一种，下面将继
续介绍更多的分组方法。
(1) 将列表、数组、Series 或索引作为分组键。分组键可以是长度与 DataFrame 匹配的任意
Series 或列表，例如：
```python
In[25]: L = [0, 1, 0, 1, 2, 0]
print(df); print(df.groupby(L).sum())
df df.groupby(L).sum()
 key data1 data2 data1 data2
0 A 0 5 0 7 17
1 B 1 0 1 4 3
2 C 2 3 2 4 7
3 A 3 3
4 B 4 7
5 C 5 9
```
因此，还有一种比前面直接用列名更啰嗦的表示方法 df.groupby('key')：
```python
In[26]: print(df); print(df.groupby(df['key']).sum())
df df.groupby(df['key']).sum()
 key data1 data2 data1 data2
0 A 0 5 A 3 8
1 B 1 0 B 5 7
2 C 2 3 C 7 12
3 A 3 3
4 B 4 7
5 C 5 9
```
(2) 用字典或 Series 将索引映射到分组名称。另一种方法是提供一个字典，将索引映射到
分组键：
```python
In[27]: df2 = df.set_index('key')
 mapping = {'A': 'vowel', 'B': 'consonant', 'C': 'consonant'}
 print(df2); print(df2.groupby(mapping).sum())
df2 df2.groupby(mapping).sum()
key data1 data2 data1 data2
A 0 5 consonant 12 19
B 1 0 vowel 3 8
C 2 3
A 3 3
B 4 7
C 5 9
```
(3) 任意 Python 函数。与前面的字典映射类似，你可以将任意 Python 函数传入 groupby，
函数映射到索引，然后新的分组输出：
```python
In[28]: print(df2); print(df2.groupby(str.lower).mean())
df2 df2.groupby(str.lower).mean()
key data1 data2 data1 data2
A 0 5 a 1.5 4.0
B 1 0 b 2.5 3.5
C 2 3 c 3.5 6.0
A 3 3
B 4 7
C 5 9
```
(4) 多个有效键构成的列表。此外，任意之前有效的键都可以组合起来进行分组，从而返回
一个多级索引的分组结果：
```python
In[29]: df2.groupby([str.lower, mapping]).mean()
Out[29]: data1 data2
 a vowel 1.5 4.0
 b consonant 2.5 3.5
 c consonant 3.5 6.0
 ```
#### 5. 分组案例
通过下例中的几行 Python 代码，我们就可以运用上述知识，获取不同方法和不同年份发现
的行星数量：
```python
In[30]: decade = 10 * (planets['year'] // 10)
 decade = decade.astype(str) + 's'
 decade.name = 'decade'
 planets.groupby(['method', decade])['number'].sum().unstack().fillna(0)
Out[30]: decade 1980s 1990s 2000s 2010s
 method
 Astrometry 0.0 0.0 0.0 2.0
 Eclipse Timing Variations 0.0 0.0 5.0 10.0
 Imaging 0.0 0.0 29.0 21.0
 Microlensing 0.0 0.0 12.0 15.0
 Orbital Brightness Modulation 0.0 0.0 0.0 5.0
 Pulsar Timing 0.0 9.0 1.0 1.0
 Pulsation Timing Variations 0.0 0.0 1.0 0.0 
 Radial Velocity 1.0 52.0 475.0 424.0
 Transit 0.0 0.0 64.0 712.0
 Transit Timing Variations 0.0 0.0 0.0 9.0
 ```
此例足以展现 GroupBy 在探索真实数据集时快速组合多种操作的能力——只用寥寥几行代
码，就可以让我们立即对过去几十年里不同年代的行星发现方法有一个大概的了解。
我建议你花点时间分析这几行代码，确保自己真正理解了每一行代码对结果产生了怎样的
影响。虽然这个例子的确有点儿复杂，但是理解这几行代码的含义可以帮你掌握分析类似
数据的方法。
### 　数据透视表
我们已经介绍过 GroupBy 抽象类是如何探索数据集内部的关联性的了。数据透视表（pivot
table）是一种类似的操作方法，常见于 Excel 与类似的表格应用中。数据透视表将每一列
数据作为输入，输出将数据不断细分成多个维度累计信息的二维数据表。人们有时容易弄
混数据透视表与 GroupBy，但我觉得数据透视表更像是一种多维的 GroupBy 累计操作。也就
是说，虽然你也可以分割 - 应用 - 组合，但是分割与组合不是发生在一维索引上，而是在
二维网格上（行列同时分组）。
### 　演示数据透视表
这一节的示例将采用泰坦尼克号的乘客信息数据库来演示，可以在 Seaborn 程序库（详情
请参见 4.16 节）获取：
```python
In[1]: import numpy as np
 import pandas as pd
 import seaborn as sns
 titanic = sns.load_dataset('titanic')
In[2]: titanic.head()
Out[2]:
 survived pclass sex age sibsp parch fare embarked class \\
0 0 3 male 22.0 1 0 7.2500 S Third
1 1 1 female 38.0 1 0 71.2833 C First
2 1 3 female 26.0 0 0 7.9250 S Third
3 1 1 female 35.0 1 0 53.1000 S First
4 0 3 male 35.0 0 0 8.0500 S Third
 who adult_male deck embark_town alive alone
0 man True NaN Southampton no False
1 woman False C Cherbourg yes False
2 woman False NaN Southampton yes True
3 woman False C Southampton yes False
4 man True NaN Southampton no True
```
这份数据包含了惨遭厄运的每位乘客的大量信息，包括性别（gender）、年龄（age）、船舱
等级（class）和船票价格（fare paid）等。

### 手工制作数据透视表
在研究这些数据之前，先将它们按照性别、最终生还状态或其他组合属性进行分组。如果
你看过前面的章节，你可能会用 GroupBy 来实现，例如这样统计不同性别乘客的生还率：
```python
In[3]: titanic.groupby('sex')[['survived']].mean()
Out[3]: survived
 sex
 female 0.742038
 male 0.188908
 ```
这组数据会立刻给我们一个直观感受：总体来说，有四分之三的女性被救，但只有五分之
一的男性被救！
这组数据很有用，但是我们可能还想进一步探索，同时观察不同性别与船舱等级的生还情
况。根据 GroupBy 的操作流程，我们也许能够实现想要的结果：将船舱等级（'class'）与
性别（'sex'）分组，然后选择生还状态（'survived'）列，应用均值（'mean'）累计函
数，再将各组结果组合，最后通过行索引转列索引操作将最里层的行索引转换成列索引，
形成二维数组。代码如下所示：
```python
In[4]: titanic.groupby(['sex', 'class'])['survived'].aggregate('mean').unstack()
Out[4]: class First Second Third
 sex
 female 0.968085 0.921053 0.500000
 male 0.368852 0.157407 0.135447
 ```
虽然这样就可以更清晰地观察乘客性别、船舱等级对其是否生还的影响，但是代码看上去
有点复杂。尽管这个管道命令的每一步都是前面介绍过的，但是要理解这个长长的语句可
不是那么容易的事。由于二维的 GroupBy 应用场景非常普遍，因此 Pandas 提供了一个快捷
方式 pivot_table 来快速解决多维的累计分析任务。
### 数据透视表语法
用 DataFrame 的 pivot_table 实现的效果等同于上一节的管道命令的代码：
```python
In[5]: titanic.pivot_table('survived', index='sex', columns='class')
Out[5]: class First Second Third
 sex
 female 0.968085 0.921053 0.500000
 male 0.368852 0.157407 0.135447
 ```
与 GroupBy 方法相比，这行代码可读性更强，而且取得的结果也一样。可能与你对 20 世纪
初的那场灾难的猜想一致，生还率最高的是船舱等级高的女性。一等舱的女性乘客基本全
部生还（露丝自然得救），而三等舱男性乘客的生还率仅为十分之一（杰克为爱牺牲）。
#### 1. 多级数据透视表
与 GroupBy 类似，数据透视表中的分组也可以通过各种参数指定多个等级。例如，我们
可能想把年龄（'age'）也加进去作为第三个维度，这就可以通过 pd.cut 函数将年龄进
行分段：
```python
In[6]: age = pd.cut(titanic['age'], [0, 18, 80])
 titanic.pivot_table('survived', ['sex', age], 'class')
Out[6]: class First Second Third
 sex age
 female (0, 18] 0.909091 1.000000 0.511628
 (18, 80] 0.972973 0.900000 0.423729
 male (0, 18] 0.800000 0.600000 0.215686
 (18, 80] 0.375000 0.071429 0.133663
 ```
对某一列也可以使用同样的策略——让我们用 pd.qcut 将船票价格按照计数项等分为两份，
加入数据透视表看看：
```python
In[7]: fare = pd.qcut(titanic['fare'], 2)
 titanic.pivot_table('survived', ['sex', age], [fare, 'class'])
Out[7]:
fare [0, 14.454]
class First Second Third \\
sex age
female (0, 18] NaN 1.000000 0.714286
 (18, 80] NaN 0.880000 0.444444
male (0, 18] NaN 0.000000 0.260870
 (18, 80] 0.0 0.098039 0.125000
fare (14.454, 512.329]
class First Second Third
sex age
female (0, 18] 0.909091 1.000000 0.318182
 (18, 80] 0.972973 0.914286 0.391304
male (0, 18] 0.800000 0.818182 0.178571
 (18, 80] 0.391304 0.030303 0.192308
 ```
结果是一个带层级索引（详情请参见 3.6 节）的四维累计数据表，通过网格显示不同数值
之间的相关性。
#### 2. 其他数据透视表选项
DataFrame 的 pivot_table 方法的完整签名如下所示：
```python
# Pandas 0.18版的函数签名
DataFrame.pivot_table(data, values=None, index=None, columns=None,
 aggfunc='mean', fill_value=None, margins=False,
 dropna=True, margins_name='All')
 ```
我们已经介绍过前面三个参数了，现在来看看其他参数。fill_value 和 dropna 这两个参数
用于处理缺失值，用法很简单，我们将在后面的示例中演示其用法。
aggfunc 参数用于设置累计函数类型，默认值是均值（mean）。与 GroupBy 的用法一样，累
计函数可以用一些常见的字符串（'sum'、'mean'、'count'、'min'、'max' 等）表示，也
可以用标准的累计函数（np.sum()、min()、sum() 等）表示。另外，还可以通过字典为不

同的列指定不同的累计函数：
```python
In[8]: titanic.pivot_table(index='sex', columns='class',
 aggfunc={'survived':sum, 'fare':'mean'})
Out[8]: fare survived
 class First Second Third First Second Third
 sex
 female 106.125798 21.970121 16.118810 91.0 70.0 72.0
 male 67.226127 19.741782 12.661633 45.0 17.0 47.0
 ```
需要注意的是，这里忽略了一个参数 values。当我们为 aggfunc 指定映射关系的时候，待
透视的数值就已经确定了。
当需要计算每一组的总数时，可以通过 margins 参数来设置：
```python
In[9]: titanic.pivot_table('survived', index='sex', columns='class', margins=True)
Out[9]: class First Second Third All
 sex
 female 0.968085 0.921053 0.500000 0.742038
 male 0.368852 0.157407 0.135447 0.188908
 All 0.629630 0.472826 0.242363 0.383838
 ```
这样就可以自动获取不同性别下船舱等级与生还率的相关信息、不同船舱等级下性别与生
还率的相关信息，以及全部乘客的生还率为 38%。margin 的标签可以通过 margins_name 参
数进行自定义，默认值是 "All"。
### 案例：美国人的生日
再来看一个有趣的例子——由美国疾病防治中心（Centers for Disease Control，CDC）提供
的公开生日数据，这些数据可以从 https://raw.githubusercontent.com/jakevdp/data-CDCbirths/
master/births.csv 下载。（Andrew Gelman 和他的团队已经对这个数据集进行了深入的分析，
详情请参见博文 http://bit.ly/2fZzW8K。）
```python
In[10]:
# shell下载数据
# !curl -O https://raw.githubusercontent.com/jakevdp/data-CDCbirths/
# master/births.csv
In[11]: births = pd.read_csv('births.csv')
```
只简单浏览一下，就会发现这些数据比较简单，只包含了不同出生日期（年月日）与性别
的出生人数：
```python
In[12]: births.head()
Out[12]: year month day gender births
 0 1969 1 1 F 4046
 1 1969 1 1 M 4440
 2 1969 1 2 F 4454
 3 1969 1 2 M 4548
 4 1969 1 3 F 4548
```
可以用一个数据透视表来探索这份数据。先增加一列表示不同年代，看看各年代的男女出
生比例：
```python
In[13]:
births['decade'] = 10 * (births['year'] // 10)
births.pivot_table('births', index='decade', columns='gender', aggfunc='sum')
Out[13]: gender F M
 decade
 1960 1753634 1846572
 1970 16263075 17121550
 1980 18310351 19243452
 1990 19479454 20420553
 2000 18229309 19106428
 ```
我们马上就会发现，每个年代的男性出生率都比女性出生率高。如果希望更直观地体现这
种趋势，可以用 Pandas 内置的画图功能将每一年的出生人数画出来（如图 3-2 所示，详情
请参见第 4 章中用 Matplotlib 画图的内容）：
```python
In[14]:
%matplotlib inline
import matplotlib.pyplot as plt
sns.set() # 使用Seaborn风格
births.pivot_table('births', index='year', columns='gender', aggfunc='sum').plot()
plt.ylabel('total births per year');
```
借助一个简单的数据透视表和 plot() 方法，我们马上就可以发现不同性别出生率的趋势。
通过肉眼观察，得知过去 50 年间的男性出生率比女性出生率高 5%。
深入探索
虽然使用数据透视表并不是必须的，但是通过 Pandas 的这个工具可以展现一些有趣的特
征。我们必须对数据做一点儿清理工作，消除由于输错了日期而造成的异常点（如 6 月 31
号）或者是缺失值（如 1999 年 6 月）。消除这些异常的简便方法就是直接删除异常值，可
以通过更稳定的 sigma 消除法（sigma-clipping，按照正态分布标准差划定范围，SciPy 中
默认是四个标准差）操作来实现：
```python
In[15]: quartiles = np.percentile(births['births'], [25, 50, 75])
 mu = quartiles[1]
 sig = 0.74 * (quartiles[2] - quartiles[0])
 ```
最后一行是样本均值的稳定性估计（robust estimate），其中 0.74 是指标准正态分布的分位
数间距。在 query() 方法（详情请参见 3.13 节）中用这个范围就可以将有效的生日数据筛
选出来了：
```python
In[16]:
births = births.query('(births > @mu - 5 * @sig) & (births < @mu + 5 * @sig)')
```
然后，将 day 列设置为整数。这列数据在筛选之前是字符串，因为数据集中有的列含有缺
失值 'null'：
```python
In[17]: # 将'day'列设置为整数。由于其中含有缺失值null，因此是字符串
 births['day'] = births['day'].astype(int)
 ```
现在就可以将年月日组合起来创建一个日期索引了（详情请参见 3.12 节），这样就可以快
速计算每一行是星期几：
```python
In[18]: # 从年月日创建一个日期索引
 births.index = pd.to_datetime(10000 * births.year +
 100 * births.month +
 births.day, format='%Y%m%d')
 births['dayofweek'] = births.index.dayofweek
 ```
用这个索引可以画出不同年代不同星期的日均出生数据（如图 3-3 所示）：
```python
In[19]:
import matplotlib.pyplot as plt
import matplotlib as mpl
births.pivot_table('births', index='dayofweek',
 columns='decade', aggfunc='mean').plot()
plt.gca().set_xticklabels(['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun'])
plt.ylabel('mean births by day');
```
注 3：你可以从我与 Željko Ivezi、Andrew J. Connolly 以及 Alexander Gray 合著的，由普林斯顿大学出版社
于 2014 年出版的 Statistics, Data Mining, and Machine Learning in Astronomy: A Practical Python Guide
for the Analysis of Survey Data 一书中了解更多关于 sigma 消除法操作的内容。


另一个有趣的图表是画出各个年份平均每天的出生人数，可以按照月和日两个维度分别对
数据进行分组：
In[20]:
births_by_date = births.pivot_table('births',
 [births.index.month, births.index.day])
births_by_date.head()
Out[20]: 1 1 4009.225
 2 4247.400
 3 4500.900
 4 4571.350
 5 4603.625
 Name: births, dtype: float64
这是一个包含月和日的多级索引。为了让数据可以用图形表示，我们可以虚构一个年份，
与月和日组合成新索引（注意日期为 2 月 29 日时，索引年份需要用闰年，例如 2012）：
```python
In[21]: births_by_date.index = [pd.datetime(2012, month, day)
 for (month, day) in births_by_date.index]
 births_by_date.head()
Out[21]: 2012-01-01 4009.225
 2012-01-02 4247.400
 2012-01-03 4500.900
 2012-01-04 4571.350
 2012-01-05 4603.625
 Name: births, dtype: float64
 ```
如果只关心月和日的话，这就是一个可以反映一年中平均每天出生人数的时间序列。可以

用 plot 方法将数据画成图（如图 3-4 所示），从图中可以看到一些有趣的趋势：
```python
In[22]: # 将结果画成图
 fig, ax = plt.subplots(figsize=(12, 4))
 births_by_date.plot(ax=ax);
```
从图中可以明显看出，在美国节假日的时候，出生人数急速下降（例如美国独立日、劳
动节、感恩节、圣诞节以及新年）。这种现象可能是由于医院放假导致的接生减少（自己
在家生），而非某种自然生育的心理学效应。关于这个趋势的更详细介绍，请参考 Andrew
Gelman 的博客（http://bit.ly/2fZzW8K）。我们将在 4.11.1 节再次使用这张图，那时将用
Matplotlib 的画图工具为这张图增加标注。
通过这个简单的案例，你会发现许多前面介绍过的 Python 和 Pandas 工具都可以相互结合，
并用于从大量数据集中获取信息。我们将在后面的章节中介绍如何用这些工具创建更复杂
的应用。
## 　向量化字符串操作
使用 Python 的一个优势就是字符串处理起来比较容易。在此基础上创建的 Pandas 同样提
供了一系列向量化字符串操作（vectorized string operation），它们都是在处理（清洗）现实
工作中的数据时不可或缺的功能。在这一节中，我们将介绍 Pandas 的字符串操作，学习如
何用它们对一个从网络采集来的杂乱无章的数据集进行局部清理。
### Pandas字符串操作简介
前面的章节已经介绍过如何用 NumPy 和 Pandas 进行一般的运算操作，因此我们也能简便
快速地对多个数组元素执行同样的操作，例如：
```python
In[1]: import numpy as np
 x = np.array([2, 3, 5, 7, 11, 13])
 x * 2
Out[1]: array([ 4, 6, 10, 14, 22, 26])
```
向量化操作简化了纯数值的数组操作语法——我们不需要再担心数组的长度或维度，只需
要关心需要的操作。然而，由于 NumPy 并没有为字符串数组提供简单的接口，因此需要
通过繁琐的 for 循环来解决问题：
```python
In[2]: data = ['peter', 'Paul', 'MARY', 'gUIDO']
 [s.capitalize() for s in data]
Out[2]: ['Peter', 'Paul', 'Mary', 'Guido']
```
虽然这么做对于某些数据可能是有效的，但是假如数据中出现了缺失值，那么这样做就会
引起异常，例如：
```python
In[3]: data = ['peter', 'Paul', None, 'MARY', 'gUIDO']
 [s.capitalize() for s in data]
---------------------------------------------------------------------------
---------------------------------------------------------------------------
AttributeError Traceback (most recent call last)
<ipython-input-3-fc1d891ab539> in <module>()
 1 data = ['peter', 'Paul', None, 'MARY', 'gUIDO']
----> 2 [s.capitalize() for s in data]
<ipython-input-3-fc1d891ab539> in <listcomp>(.0)
 1 data = ['peter', 'Paul', None, 'MARY', 'gUIDO']
----> 2 [s.capitalize() for s in data]
AttributeError: 'NoneType' object has no attribute 'capitalize'
---------------------------------------------------------------------------
```
Pandas 为包含字符串的 Series 和 Index 对象提供的 str 属性堪称两全其美的方法，它既
可以满足向量化字符串操作的需求，又可以正确地处理缺失值。例如，我们用前面的数据
data 创建了一个 Pandas 的 Series：
```python
In[4]: import pandas as pd
 names = pd.Series(data)
 names
Out[4]: 0 peter
 1 Paul
 2 None
 3 MARY
 4 gUIDO
 dtype: object
 ```
现在就可以直接调用转换大写方法 capitalize() 将所有的字符串变成大写形式，缺失值会
被跳过：
```python
In[5]: names.str.capitalize()
Out[5]: 0 Peter
 1 Paul
 2 None 

 3 Mary
 4 Guido
 dtype: object
 ```
在 str 属性后面用 Tab 键，可以看到 Pandas 支持的所有向量化字符串方法。
 
### Pandas字符串方法列表
如果你熟悉 Python 的字符串方法的话，就会发现 Pandas 绝大多数的字符串语法都很直观，
甚至可以列成一个表格。在深入论述后面的内容之前，让我们先从这一步开始。这一节的
示例将采用一些人名来演示：
```python
In[6]: monte = pd.Series(['Graham Chapman', 'John Cleese', 'Terry Gilliam',
 'Eric Idle', 'Terry Jones', 'Michael Palin'])
 ```
#### 1. 与Python字符串方法相似的方法
几乎所有 Python 内置的字符串方法都被复制到 Pandas 的向量化字符串方法中。下面的表
格列举了 Pandas 的 str 方法借鉴 Python 字符串方法的内容：
```python
len() lower() translate() islower()
ljust() upper() startswith() isupper()
rjust() find() endswith() isnumeric()
center() rfind() isalnum() isdecimal()
zfill() index() isalpha() split()
strip() rindex() isdigit() rsplit()
rstrip() capitalize() isspace() partition()
lstrip() swapcase() istitle() rpartition()
```
需要注意的是，这些方法的返回值不同，例如 lower() 方法返回一个字符串 Series：
```python
In[7]: monte.str.lower()
Out[7]: 0 graham chapman
 1 john cleese
 2 terry gilliam
 3 eric idle
 4 terry jones
 5 michael palin
 dtype: object
 ```
但是有些方法返回数值：
```python
In[8]: monte.str.len()
Out[8]: 0 14
 1 11
 2 13
 3 9
 4 11
 5 13
 dtype: int64
```
有些方法返回布尔值：
```python
In[9]: monte.str.startswith('T')
Out[9]: 0 False
 1 False
 2 True
 3 False
 4 True
 5 False
 dtype: bool
 ```
还有些方法返回列表或其他复合值：
```python
In[10]: monte.str.split()
Out[10]: 0 [Graham, Chapman]
 1 [John, Cleese]
 2 [Terry, Gilliam]
 3 [Eric, Idle]
 4 [Terry, Jones]
 5 [Michael, Palin]
 dtype: object
 ```
在接下来的内容中，我们将进一步学习这类由列表元素构成的 Series（series-of-lists）对象。
#### 2. 使用正则表达式的方法
还有一些支持正则表达式的方法可以用来处理每个字符串元素。表 3-4 中的内容是 Pandas
向量化字符串方法根据 Python 标准库的 re 模块函数实现的 API。
表3-4：Pandas向量化字符串方法与Python标准库的re模块函数的对应关系
方法 描述
match() 对每个元素调用 re.match()，返回布尔类型值
extract() 对每个元素调用 re.match()，返回匹配的字符串组（groups）
findall() 对每个元素调用 re.findall()
replace() 用正则模式替换字符串
contains() 对每个元素调用 re.search()，返回布尔类型值
count() 计算符合正则模式的字符串的数量
split() 等价于 str.split()，支持正则表达式
rsplit() 等价于 str.rsplit()，支持正则表达式
通过这些方法，你就可以实现各种有趣的操作了。例如，可以提取元素前面的连续字母作
为每个人的名字（first name）：
```python
In[11]: monte.str.extract('([A-Za-z]+)')
Out[11]: 0 Graham
 1 John
 2 Terry
 3 Eric
 4 Terry 

 5 Michael
 dtype: object
 ```
我们还能实现更复杂的操作，例如找出所有开头和结尾都是辅音字母的名字——这可以用
正则表达式中的开始符号（^）与结尾符号（$）来实现：
```python
In[12]: monte.str.findall(r'^[^AEIOU].*[^aeiou]$')
Out[12]: 0 [Graham Chapman]
 1 []
 2 [Terry Gilliam]
 3 []
 4 [Terry Jones]
 5 [Michael Palin]
 dtype: object
 ```
能将正则表达式应用到 Series 与 DataFrame 之中的话，就有可能实现更多的数据分析与清
洗方法。
### 3. 其他字符串方法
还有其他一些方法也可以实现方便的操作（如表 3-5 所示）。
表3-5 其他Pandas字符串方法
方法 描述
get() 获取元素索引位置上的值，索引从 0 开始
slice() 对元素进行切片取值
slice_replace() 对元素进行切片替换
cat() 连接字符串（此功能比较复杂，建议阅读文档）
repeat() 重复元素
normalize() 将字符串转换为 Unicode 规范形式
pad() 在字符串的左边、右边或两边增加空格
wrap() 将字符串按照指定的宽度换行
join() 用分隔符连接 Series 的每个元素
get_dummies() 按照分隔符提取每个元素的 dummy 变量，转换为独热（one-hot）编码的 DataFrame
(1) 向量化字符串的取值与切片操作。这里需要特别指出的是，get() 与 slice() 操作可以
从每个字符串数组中获取向量化元素。例如，我们可以通过 str.slice(0, 3) 获取每个
字符串数组的前三个字符。通过 Python 的标准取值方法也可以取得同样的效果，例如
df.str.slice(0, 3) 等价于 df.str[0:3]：
```python
In[13]: monte.str[0:3]
Out[13]: 0 Gra
 1 Joh
 2 Ter
 3 Eri
 4 Ter
 5 Mic
 dtype: object
```
df.str.get(i) 与 df.str[i] 的按索引取值效果类似。
get() 与 slice() 操作还可以在 split() 操作之后使用。例如，要获取每个姓名的姓
（last name），可以结合使用 split() 与 get()：
```python
In[14]: monte.str.split().str.get(-1)
Out[14]: 0 Chapman
 1 Cleese
 2 Gilliam
 3 Idle
 4 Jones
 5 Palin
 dtype: object
 ```
(2) 指标变量。另一个需要多花点儿时间解释的是 get_dummies() 方法。当你的数据有一列
包含了若干已被编码的指标（coded indicator）时，这个方法就能派上用场了。例如，
假设有一个包含了某种编码信息的数据集，如 A= 出生在美国、B= 出生在英国、C= 喜
欢奶酪、D= 喜欢午餐肉：
```python
In[15]:
full_monte = pd.DataFrame({'name': monte,
 'info': ['B|C|D', 'B|D', 'A|C', 'B|D', 'B|C',
 'B|C|D']})
full_monte
Out[15]: info name
 0 B|C|D Graham Chapman
 1 B|D John Cleese
 2 A|C Terry Gilliam
 3 B|D Eric Idle
 4 B|C Terry Jones
 5 B|C|D Michael Palin
 ```
get_dummies() 方法可以让你快速将这些指标变量分割成一个独热编码的 DataFrame（每
个元素都是 0 或 1）：
```python
In[16]: full_monte['info'].str.get_dummies('|')
Out[16]: A B C D
 0 0 1 1 1
 1 0 1 0 1
 2 1 0 1 0
 3 0 1 0 1
 4 0 1 1 0
 5 0 1 1 1
 ```
通过 Pandas 自带的这些字符串操作方法，你就可以建立一个功能无比强大的字符串处
理程序来清洗自己的数据了。
虽然本书将不再继续介绍这些方法，但是希望你仔细阅读 Pandas 在线文档中“Working
with Text Data”（http://pandas.pydata.org/pandas-docs/stable/text.html）节，或者阅读 3.14 节
的相关资源。

### 　案例：食谱数据库
前面介绍的这些向量化字符串操作方法非常适合用来处理现实中那些凌乱的数据。下面将
通过一个从不同网站获取的公开食谱数据库的案例来进行演示。我们的目标是将这些食谱
数据解析为食材列表，这样就可以根据现有的食材快速找到食谱。
获取数据的脚本可以在 https://github.com/fictivekin/openrecipes 上找到，那里还有最新版的
数据库链接。
截至 2016 年春，这个数据集已经有 30MB 了。可以通过下面的命令下载并解压数据：
```python
In[17]: # !curl -O http://openrecipes.s3.amazonaws.com/recipeitems-latest.json.gz
 # !gunzip recipeitems-latest.json.gz
这个数据库是 JSON 格式的，来试试通过 pd.read_json 读取数据：
In[18]: try:
 recipes = pd.read_json('recipeitems-latest.json')
 except ValueError as e:
 print("ValueError:", e)
ValueError: Trailing data
```
糟糕！我们得到的竟然是提示数据里有“trailing data”（数据断行）的 ValueError 错误。
从网上搜索这个错误，得知原因好像是虽然文件中的每一行都是一个有效的 JSON 对象，
但是全文却不是这样。来看看文件是不是这样：
```python
In[19]: with open('recipeitems-latest.json') as f:
 line = f.readline()
 pd.read_json(line).shape
Out[19]: (2, 12)
```
显然每一行都是一个有效的 JSON 对象，因此需要将这些字符串连接在一起。解决这个问
题的一种方法就是新建一个字符串，将所有行 JSON 对象连接起来，然后再通过 pd.read_
json 来读取所有数据：
```python
In[20]: # 将文件内容读取成Python数组
 with open('recipeitems-latest.json', 'r') as f:
 # 提取每一行内容
 data = (line.strip() for line in f)
 # 将所有内容合并成一个列表
 data_json = "[{0}]".format(','.join(data))
 # 用JSON形式读取数据
 recipes = pd.read_json(data_json)
In[21]: recipes.shape
Out[21]: (173278, 17)
这样就会看到将近 20 万份食谱，共 17 列。抽一行看看具体内容：
In[22]: recipes.iloc[0] 

Out[22]:
_id {'$oid': '5160756b96cc62079cc2db15'}
cookTime PT30M
creator NaN
dateModified NaN
datePublished 2013-03-11
description Late Saturday afternoon, after Marlboro Man ha...
image http://static.thepioneerwoman.com/cooking/file...
ingredients Biscuits\n3 cups All-purpose Flour\n2 Tablespo...
name Drop Biscuits and Sausage Gravy
prepTime PT10M
recipeCategory NaN
recipeInstructions NaN
recipeYield 12
source thepioneerwoman
totalTime NaN
ts {'$date': 1365276011104}
url http://thepioneerwoman.com/cooking/2013/03/dro...
Name: 0, dtype: object
```
这里有一堆信息，而且其中有不少都和从网站上抓取的数据一样，字段形式混乱。值得关
注的是，食材列表是字符串形式，我们需要从中抽取感兴趣的信息。下面来仔细看看这个
字段：
```python
In[23]: recipes.ingredients.str.len().describe()
Out[23]: count 173278.000000
 mean 244.617926
 std 146.705285
 min 0.000000
 25% 147.000000
 50% 221.000000
 75% 314.000000
 max 9067.000000
 Name: ingredients, dtype: float64
 ```
食材列表平均 250 个字符，最短的字符串是 0，最长的竟然接近 1 万字符！
出于好奇心，来看看这个拥有最长食材列表的究竟是哪道菜：
```python
In[24]: recipes.name[np.argmax(recipes.ingredients.str.len())]
Out[24]: 'Carrot Pineapple Spice &amp; Brownie Layer Cake with Whipped Cream
&amp; Cream Cheese Frosting and Marzipan Carrots'
```
从名字就可以看出，这绝对是个复杂的食谱。
我们还可以再做一些累计探索，例如看看哪些食谱是早餐：
```python
In[33]: recipes.description.str.contains('[Bb]reakfast').sum()
Out[33]: 3524
```
或者看看有多少食谱用肉桂（cinnamon）作为食材：
```python
In[34]: recipes.ingredients.str.contains('[Cc]innamon').sum()
Out[34]: 10526
```
还可以看看究竟是哪些食谱里把肉桂错写成了“cinamon”：
```python
In[27]: recipes.ingredients.str.contains('[Cc]inamon').sum()
Out[27]: 11
```
这些基本的数据探索都可以用 Pandas 的字符串工具来处理，Python 非常适合进行类似的
数据清理工作。
#### 1. 制作简易的美食推荐系统
现在让我们更进一步，来制作一个简易的美食推荐系统：如果用户提供一些食材，系统
就会推荐使用了所有食材的食谱。这说起来是容易，但是由于大量不规则（heterogeneity）
数据的存在，这个任务变得十分复杂，例如并没有一个简单直接的办法可以从每一行数据
中清理出一份干净的食材列表。因此，我们在这里简化处理：首先提供一些常见食材列
表，然后通过简单搜索判断这些食材是否在食谱中。为了简化任务，这里只列举常用的香
料和调味料：
```python
In[28]: spice_list = ['salt', 'pepper', 'oregano', 'sage', 'parsley',
 'rosemary', 'tarragon', 'thyme', 'paprika', 'cumin']
 ```
现在就可以通过一个由 True 与 False 构成的布尔类型的 DataFrame 来判断食材是否出现在
某个食谱中：
```python
In[29]:
import re
spice_df = pd.DataFrame(
 dict((spice, recipes.ingredients.str.contains(spice, re.IGNORECASE))
 for spice in spice_list))
spice_df.head()
Out[29]:
 cumin oregano paprika parsley pepper rosemary sage salt tarragon thyme
0 False False False False False False True False False False
1 False False False False False False False False False False
2 True False False False True False False True False False
3 False False False False False False False False False False
4 False False False False False False False False False False
```
现在，来找一份使用了欧芹（parsley）、辣椒粉（paprika）和龙蒿叶（tarragon）这三种食
材的食谱。我们可以通过 3.13 节介绍的 DataFrame 的 query() 方法来快速完成计算：
```python
In[30]: selection = spice_df.query('parsley & paprika & tarragon')
 len(selection)
Out[30]: 10
```
最后只找到了十份同时包含这三种食材的食谱，让我们用索引看看究竟是哪些食谱：
```python
In[31]: recipes.name[selection.index]
Out[31]: 2069 All cremat with a Little Gem, dandelion and wa...
 74964 Lobster with Thermidor butter
 93768 Burton's Southern Fried Chicken with White Gravy
 113926 Mijo's Slow Cooker Shredded Beef
 137686 Asparagus Soup with Poached Eggs
 140530 Fried Oyster Po'boys
 158475 Lamb shank tagine with herb tabbouleh
 158486 Southern fried chicken in buttermilk
 163175 Fried Chicken Sliders with Pickles + Slaw
 165243 Bar Tartine Cauliflower Salad
 Name: name, dtype: object
 ```
现在已经将搜索范围缩小到了原来近两万份食谱的两千分之一了，这样就可以从这个小集
合中精挑细选出中意的食谱。
#### 2. 继续完善美食推荐系统
希望这个示例能让你对 Pandas 字符串方法可以高效解决哪些数据清理问题有个初步概念。
当然，如果要建立一个稳定的美食推荐系统，还需要做大量的工作！从每个食谱中提取完
整的食材列表是这个任务的重中之重。不过，由于食材的书写格式千奇百怪，解析它们需
要耗费大量时间。这其实也揭示了数据科学的真相——真实数据的清洗与整理工作往往会
占据的大部分时间，而使用 Pandas 提供的工具可以提高你的工作效率。
## 　处理时间序列
由于 Pandas 最初是为金融模型而创建的，因此它拥有一些功能非常强大的日期、时间、带
时间索引数据的处理工具。本节将介绍的日期与时间数据主要包含三类。
• 时间戳表示某个具体的时间点（例如 2015 年 7 月 4 日上午 7 点）。
• 时间间隔与周期表示开始时间点与结束时间点之间的时间长度，例如 2015 年（指的是
2015 年 1 月 1 日至 2015 年 12 月 31 日这段时间间隔）。周期通常是指一种特殊形式的
时间间隔，每个间隔长度相同，彼此之间不会重叠（例如，以 24 小时为周期构成每一天）。
• 时间增量（time delta）或持续时间（duration）表示精确的时间长度（例如，某程序运
行持续时间 22.56 秒）。
在本节内容中，我们将介绍 Pandas 中的 3 种日期 / 时间数据类型的具体用法。由于篇幅有
限，后文无法对 Python 或 Pandas 的时间序列工具进行详细的介绍，仅仅是通过一个宽泛
的综述，总结何时应该使用它们。在开始介绍 Pandas 的时间序列工具之前，我们先简单介
绍一下 Python 处理日期与时间数据的工具。在介绍完一些值得深入学习的资源之后，再通
过一些简短的示例来演示 Pandas 处理时间序列数据的方法。
###  Python的日期与时间工具
在 Python 标准库与第三方库中有许多可以表示日期、时间、时间增量和时间跨度
（timespan）的工具。尽管 Pandas 提供的时间序列工具更适合用来处理数据科学问题，但是
了解 Pandas 与 Python 标准库以及第三方库中的其他时间序列工具之间的关联性将大有裨益。
#### 1. 原生Python的日期与时间工具：datetime与dateutil
Python 基本的日期与时间功能都在标准库的 datetime 模块中。如果和第三方库 dateutil
模块搭配使用，可以快速实现许多处理日期与时间的功能。例如，你可以用 datetime 类型
创建一个日期：
```python
In[1]: from datetime import datetime
 datetime(year=2015, month=7, day=4)
Out[1]: datetime.datetime(2015, 7, 4, 0, 0)
```
或者使用 dateutil 模块对各种字符串格式的日期进行正确解析：
```python
In[2]: from dateutil import parser
 date = parser.parse("4th of July, 2015")
 date
Out[2]: datetime.datetime(2015, 7, 4, 0, 0)
```
一旦有了 datetime 对象，就可以进行许多操作了，例如打印出这一天是星期几：
```python
In[3]: date.strftime('%A')
Out[3]: 'Saturday'
```
在最后一行代码中，为了打印出是星期几，我们使用了一个标准字符串格式（standard
string format）代码 "%A"，你可以在 Python 的 datetime 文档（https://docs.python.org/3/library/
datetime.html）的“strftime”节（https://docs.python.org/3/library/datetime.html#strftime-andstrptime-behavior）查看具体信息。关于 dateutil 的其他日期功能可以通过 dateutil 的在线文
档（http://labix.org/python-dateutil）学习。还有一个值得关注的程序包是 pytz（http://pytz.
sourceforge.net/），这个工具解决了绝大多数时间序列数据都会遇到的难题：时区。
datetime 和 dateutil 模块在灵活性与易用性方面都表现出色，你可以用这些对象及其相
应的方法轻松完成你感兴趣的任意操作。但如果你处理的时间数据量比较大，那么速度
就会比较慢。就像之前介绍过的 Python 的原生列表对象没有 NumPy 中已经被编码的数值
类型数组的性能好一样，Python 的原生日期对象同样也没有 NumPy 中已经被编码的日期
（encoded dates）类型数组的性能好。
#### 2. 时间类型数组：NumPy的datetime64类型
Python 原生日期格式的性能弱点促使 NumPy 团队为 NumPy 增加了自己的时间序列类型。
datetime64 类型将日期编码为 64 位整数，这样可以让日期数组非常紧凑（节省内存）。
datetime64 需要在设置日期时确定具体的输入类型：
```python
In[4]: import numpy as np
 date = np.array('2015-07-04', dtype=np.datetime64)
 date
Out[4]: array(datetime.date(2015, 7, 4), dtype='datetime64[D]')
```
但只要有了这个日期格式，就可以进行快速的向量化运算：
```python
In[5]: date + np.arange(12)
Out[5]:
array(['2015-07-04', '2015-07-05', '2015-07-06', '2015-07-07',
 '2015-07-08', '2015-07-09', '2015-07-10', '2015-07-11',
 '2015-07-12', '2015-07-13', '2015-07-14', '2015-07-15'],
 dtype='datetime64[D]')
 ```
因为 NumPy 的 datetime64 数组内元素的类型是统一的，所以这种数组的运算速度会比
Python 的 datetime 对象的运算速度快很多，尤其是在处理较大数组时（关于向量化运算的
内容已经在 2.3 节介绍过）。
datetime64 与 timedelta64 对 象 的 一 个 共 同 特 点 是， 它 们 都 是 在 基本时间单位
（fundamental time unit）的基础上建立的。由于 datetime64 对象是 64 位精度，所以可编码
的时间范围可以是基本单元的 264 倍。也就是说，datetime64 在时间精度（time resolution）
与最大时间跨度（maximum time span）之间达成了一种平衡。
比如你想要一个时间纳秒（nanosecond，ns）级的时间精度，那么你就可以将时间编码到
0~264 纳秒或 600 年之内，NumPy 会自动判断输入时间需要使用的时间单位。例如，下面
是一个以天为单位的日期：
```python
In[6]: np.datetime64('2015-07-04')
Out[6]: numpy.datetime64('2015-07-04')
```
而这是一个以分钟为单位的日期：
```python
In[7]: np.datetime64('2015-07-04 12:00')
Out[7]: numpy.datetime64('2015-07-04T12:00')
```
需要注意的是，时区将自动设置为执行代码的操作系统的当地时区。你可以通过各种格式
的代码设置基本时间单位。例如，将时间单位设置为纳秒：
```python
In[8]: np.datetime64('2015-07-04 12:59:59.50', 'ns')
Out[8]: numpy.datetime64('2015-07-04T12:59:59.500000000')
```
NumPy 的 datetime64 文档（http://docs.scipy.org/doc/numpy/reference/arrays.datetime.html）
总结了所有支持相对与绝对时间跨度的时间与日期单位格式代码，表 3-6 对此总结如下。
表3-6：日期与时间单位格式代码
代码 含义 时间跨度 (相对) 时间跨度 (绝对)
Y 年（year） ± 9.2e18 年 [9.2e18 BC, 9.2e18 AD]
M 月（month） ± 7.6e17 年 [7.6e17 BC, 7.6e17 AD]
W 周（week） ± 1.7e17 年 [1.7e17 BC, 1.7e17 AD]
D 日（day） ± 2.5e16 年 [2.5e16 BC, 2.5e16 AD]
h 时（hour） ± 1.0e15 年 [1.0e15 BC, 1.0e15 AD]
m 分（minute） ± 1.7e13 年 [1.7e13 BC, 1.7e13 AD]
s 秒（second） ± 2.9e12 年 [ 2.9e9 BC, 2.9e9 AD]

代码 含义 时间跨度 (相对) 时间跨度 (绝对)
ms 毫秒（millisecond） ± 2.9e9 年 [ 2.9e6 BC, 2.9e6 AD]
us 微秒（microsecond） ± 2.9e6 年 [290301 BC, 294241 AD]
ns 纳秒（nanosecond） ± 292 年 [ 1678 AD, 2262 AD]
ps 皮秒（picosecond） ± 106 天 [ 1969 AD, 1970 AD]
fs 飞秒（femtosecond） ± 2.6 小时 [ 1969 AD, 1970 AD]
as 原秒（attosecond） ± 9.2 秒 [ 1969 AD, 1970 AD]
对于日常工作中的时间数据类型，默认单位都用纳秒 datetime64[ns]，因为用它来表示时
间范围精度可以满足绝大部分需求。
最后还需要说明一点，虽然 datetime64 弥补了 Python 原生的 datetime 类型的不足，但它
缺少了许多 datetime（尤其是 dateutil）原本具备的便捷方法与函数，具体内容请参考
NumPy 的 datetime64 文档（http://docs.scipy.org/doc/numpy/reference/arrays.datetime.html）。
#### 3. Pandas的日期与时间工具：理想与现实的最佳解决方案
Pandas 所有关于日期与时间的处理方法全部都是通过 Timestamp 对象实现的，它利用
numpy.datetime64 的有效存储和向量化接口将 datetime 和 dateutil 的易用性有机结合起
来。Pandas 通过一组 Timestamp 对象就可以创建一个可以作为 Series 或 DataFrame 索引的
DatetimeIndex，我们将在后面介绍许多类似的例子。
例如，可以用 Pandas 的方式演示前面介绍的日期与时间功能。我们可以灵活处理不同格式
的日期与时间字符串，获取某一天是星期几：
```python
In[9]: import pandas as pd
 date = pd.to_datetime("4th of July, 2015")
 date
Out[9]: Timestamp('2015-07-04 00:00:00')
In[10]: date.strftime('%A')
Out[10]: 'Saturday'
```
另外，也可以直接进行 NumPy 类型的向量化运算：
```python
In[11]: date + pd.to_timedelta(np.arange(12), 'D')
Out[11]: DatetimeIndex(['2015-07-04', '2015-07-05', '2015-07-06', '2015-07-07',
 '2015-07-08', '2015-07-09', '2015-07-10', '2015-07-11',
 '2015-07-12', '2015-07-13', '2015-07-14', '2015-07-15'],
 dtype='datetime64[ns]', freq=None)
 ```
下面将详细介绍 Pandas 用来处理时间序列数据的工具。
### Pandas时间序列：用时间作索引
Pandas 时间序列工具非常适合用来处理带时间戳的索引数据。例如，我们可以通过一个时
间索引数据创建一个 Series 对象：
```python
In[12]: index = pd.DatetimeIndex(['2014-07-04', '2014-08-04',
 '2015-07-04', '2015-08-04'])
 data = pd.Series([0, 1, 2, 3], index=index)
 data
Out[12]: 2014-07-04 0
 2014-08-04 1
 2015-07-04 2
 2015-08-04 3
 dtype: int64
 ```
有了一个带时间索引的 Series 之后，就能用它来演示之前介绍过的 Series 取值方法，可
以直接用日期进行切片取值：
```python
In[13]: data['2014-07-04':'2015-07-04']
Out[13]: 2014-07-04 0
 2014-08-04 1
 2015-07-04 2
 dtype: int64
 ```
另外，还有一些仅在此类 Series 上可用的取值操作，例如直接通过年份切片获取该年的
数据：
```python
In[14]: data['2015']
Out[14]: 2015-07-04 2
 2015-08-04 3
 dtype: int64
 ```
下面将介绍一些示例，体现将日期作为索引为运算带来的便利性。在此之前，让我们仔细
看看现有的时间序列数据结构。
### Pandas时间序列数据结构
本节将介绍 Pandas 用来处理时间序列的基础数据类型。
• 针对时间戳数据，Pandas 提供了 Timestamp 类型。与前面介绍的一样，它本质上是
Python 的原生 datetime 类型的替代品，但是在性能更好的 numpy.datetime64 类型的基
础上创建。对应的索引数据结构是 DatetimeIndex。
• 针对时间周期数据，Pandas 提供了 Period 类型。这是利用 numpy.datetime64 类型将固
定频率的时间间隔进行编码。对应的索引数据结构是 PeriodIndex。
• 针对时间增量或持续时间，Pandas 提供了 Timedelta 类型。Timedelta 是一种代替 Python
原生 datetime.timedelta 类型的高性能数据结构，同样是基于 numpy.timedelta64 类型。
对应的索引数据结构是 TimedeltaIndex。
最基础的日期 / 时间对象是 Timestamp 和 DatetimeIndex。这两种对象可以直接使用，最常用
的方法是 pd.to_datetime() 函数，它可以解析许多日期与时间格式。对 pd.to_datetime() 传
递一个日期会返回一个 Timestamp 类型，传递一个时间序列会返回一个 DatetimeIndex 类型：
```python
In[15]: dates = pd.to_datetime([datetime(2015, 7, 3), '4th of July, 2015',
 '2015-Jul-6', '07-07-2015', '20150708'])
 dates
Out[15]: DatetimeIndex(['2015-07-03', '2015-07-04', '2015-07-06', '2015-07-07',
 '2015-07-08'],
 dtype='datetime64[ns]', freq=None)
 ```
任何 DatetimeIndex 类型都可以通过 to_period() 方法和一个频率代码转换成 PeriodIndex
类型。下面用 'D' 将数据转换成单日的时间序列：
```python
In[16]: dates.to_period('D')
Out[16]: PeriodIndex(['2015-07-03', '2015-07-04', '2015-07-06', '2015-07-07',
 '2015-07-08'],
 dtype='int64', freq='D')
 ```
当用一个日期减去另一个日期时，返回的结果是 TimedeltaIndex 类型：
```python
In[17]: dates - dates[0]
Out[17]:
TimedeltaIndex(['0 days', '1 days', '3 days', '4 days', '5 days'],
 dtype='timedelta64[ns]', freq=None)
 ```
有规律的时间序列：pd.date_range()
为了能更简便地创建有规律的时间序列，Pandas 提供了一些方法：pd.date_range() 可以
处理时间戳、pd.period_range() 可以处理周期、pd.timedelta_range() 可以处理时间间
隔。我们已经介绍过，Python 的 range() 和 NumPy 的 np.arange() 可以用起点、终点和步
长（可选的）创建一个序列。pd.date_range() 与之类似，通过开始日期、结束日期和频率
代码（同样是可选的）创建一个有规律的日期序列，默认的频率是天：
```python
In[18]: pd.date_range('2015-07-03', '2015-07-10')
Out[18]: DatetimeIndex(['2015-07-03', '2015-07-04', '2015-07-05', '2015-07-06',
 '2015-07-07', '2015-07-08', '2015-07-09', '2015-07-10'],
 dtype='datetime64[ns]', freq='D')
 ```
此外，日期范围不一定非是开始时间与结束时间，也可以是开始时间与周期数 periods：
```python
In[19]: pd.date_range('2015-07-03', periods=8)
Out[19]: DatetimeIndex(['2015-07-03', '2015-07-04', '2015-07-05', '2015-07-06',
 '2015-07-07', '2015-07-08', '2015-07-09', '2015-07-10'],
 dtype='datetime64[ns]', freq='D')
 ```
你可以通过 freq 参数改变时间间隔，默认值是 D。例如，可以创建一个按小时变化的时间戳：
```python
In[20]: pd.date_range('2015-07-03', periods=8, freq='H')
Out[20]: DatetimeIndex(['2015-07-03 00:00:00', '2015-07-03 01:00:00',
 '2015-07-03 02:00:00', '2015-07-03 03:00:00',
 '2015-07-03 04:00:00', '2015-07-03 05:00:00',
 '2015-07-03 06:00:00', '2015-07-03 07:00:00'],
 dtype='datetime64[ns]', freq='H')
```
如果要创建一个有规律的周期或时间间隔序列，有类似的函数 pd.period_range() 和
pd.timedelta_range()。下面是一个以月为周期的示例：
```python
In[21]: pd.period_range('2015-07', periods=8, freq='M')
Out[21]:
PeriodIndex(['2015-07', '2015-08', '2015-09', '2015-10', '2015-11', '2015-12',
 '2016-01', '2016-02'],
 dtype='int64', freq='M')
 ```
以及一个以小时递增的序列：
```python
In[22]: pd.timedelta_range(0, periods=10, freq='H')
Out[22]:
TimedeltaIndex(['00:00:00', '01:00:00', '02:00:00', '03:00:00', '04:00:00',
 '05:00:00', '06:00:00', '07:00:00', '08:00:00', '09:00:00'],
 dtype='timedelta64[ns]', freq='H')
 ```
掌握 Pandas 频率代码是使用所有这些时间序列创建方法的必要条件。接下来，我们将总结
这些代码。
### 　时间频率与偏移量
Pandas 时间序列工具的基础是时间频率或偏移量（offset）代码。就像之前见过的 D（day）
和 H（hour）代码，我们可以用这些代码设置任意需要的时间间隔。表 3-7 总结了主要的频
率代码。
表3-7：Pandas频率代码
代码 描述 代码 描述
D 天（calendar day，按日历算，含双休日） B 天（business day，仅含工作日）
W 周（weekly）
M 月末（month end） BM 月末（business month end，仅含工作日）
Q 季末（quarter end） BQ 季末（business quarter end，仅含工作日）
A 年末（year end） BA 年末（business year end，仅含工作日）
H 小时（hours） BH 小时（business hours，工作时间）
T 分钟（minutes）
S 秒（seconds）
L 毫秒（milliseonds）
U 微秒（microseconds）
N 纳秒（nanoseconds）
月、季、年频率都是具体周期的结束时间（月末、季末、年末），而有一些以 S（start，开
始）为后缀的代码表示日期开始（如表 3-8 所示）。
Pandas数据处理 ｜ 173
表3-8：带开始索引的频率代码
代码 频率
MS 月初（month start）
BMS 月初（business month start，仅含工作日）
QS 季初（quarter start）
BQS 季初（business quarter start，仅含工作日）
AS 年初（year start）
BAS 年初（business year start，仅含工作日）
另外，你可以在频率代码后面加三位月份缩写字母来改变季、年频率的开始时间。
• Q-JAN、BQ-FEB、QS-MAR、BQS-APR 等。
• A-JAN、BA-FEB、AS-MAR、BAS-APR 等。
同理，也可以在后面加三位星期缩写字母来改变一周的开始时间。
• W-SUN、W-MON、W-TUE、W-WED 等。
在这些代码的基础上，还可以将频率组合起来创建的新的周期。例如，可以用小时（H）
和分钟（T）的组合来实现 2 小时 30 分钟：
```python
In[23]: pd.timedelta_range(0, periods=9, freq="2H30T")
Out[23]:
TimedeltaIndex(['00:00:00', '02:30:00', '05:00:00', '07:30:00', '10:00:00',
 '12:30:00', '15:00:00', '17:30:00', '20:00:00'],
 dtype='timedelta64[ns]', freq='150T')
 ```
所有这些频率代码都对应 Pandas 时间序列的偏移量，具体内容可以在 pd.tseries.offsets
模块中找到。例如，可以用下面的方法直接创建一个工作日偏移序列：
```python
In[24]: from pandas.tseries.offsets import BDay
 pd.date_range('2015-07-01', periods=5, freq=BDay())
Out[24]: DatetimeIndex(['2015-07-01', '2015-07-02', '2015-07-03', '2015-07-06',
 '2015-07-07'],
 dtype='datetime64[ns]', freq='B')
 ```
关于时间频率与偏移量的更多内容，请参考 Pandas 在线文档“Date Offset objects”（http://
pandas.pydata.org/pandas-docs/stable/timeseries.html#dateoffset-objects）节。
### 　重新取样、迁移和窗口
用日期和时间直观地组织与获取数据是 Pandas 时间序列工具最重要的功能之一。Pandas
不仅支持普通索引功能（合并数据时自动索引对齐、直观的数据切片和取值方法等），还
专为时间序列提供了额外的操作。
下面让我们用一些股票数据来演示这些功能。由于 Pandas 最初是为金融数据模型服务
的，因此可以用它非常方便地获取金融数据。例如，pandas-datareader 程序包（可以通
过 conda install pandas-datareader 进行安装）知道如何从一些可用的数据源导入金融数
据，包含 Yahoo 财经、Google 财经和其他数据源。下面来导入 Google 的历史股票价格：
```python
In[25]: from pandas_datareader import data
 goog = data.DataReader('GOOG', start='2004', end='2016',
 data_source='google')
 goog.head()
Out[25]: Open High Low Close Volume
 Date
 2004-08-19 49.96 51.98 47.93 50.12 NaN
 2004-08-20 50.69 54.49 50.20 54.10 NaN
 2004-08-23 55.32 56.68 54.47 54.65 NaN
 2004-08-24 55.56 55.74 51.73 52.38 NaN
 2004-08-25 52.43 53.95 51.89 52.95 NaN
 ```
出于简化的目的，这里只用收盘价：
```python
In[26]: goog = goog['Close']
设置 Matplotlib 之后，就可以通过 plot() 画出可视化图了（如图 3-5 所示）：
In[27]: %matplotlib inline
 import matplotlib.pyplot as plt
 import seaborn; seaborn.set()
In[28]: goog.plot();
```

#### 1. 重新取样与频率转换
处理时间序列数据时，经常需要按照新的频率（更高频率、更低频率）对数据进行重新
取样。你可以通过 resample() 方法解决这个问题，或者用更简单的 asfreq() 方法。这两
个方法的主要差异在于，resample() 方法是以数据累计（data aggregation）为基础，而
asfreq() 方法是以数据选择（data selection）为基础。
看到 Google 的收盘价之后，让我们用两种方法对数据进行向后取样（down-sample）。这里
用年末（'BA'，最后一个工作日）对数据进行重新取样（如图 3-6 所示）：
```python
In[29]: goog.plot(alpha=0.5, style='-')
 goog.resample('BA').mean().plot(style=':')
 goog.asfreq('BA').plot(style='--');
 plt.legend(['input', 'resample', 'asfreq'],
 loc='upper left');
```
请注意这两种取样方法的差异：在每个数据点上，resample 反映的是上一年的均值，而
asfreq 反映的是上一年最后一个工作日的收盘价。
在进行向前取样（up-sampling）时，resample() 与 asfreq() 的用法大体相同，不过重新取
样有许多种配置方式。操作时，两种方法都默认将向前取样作为缺失值处理，也就是说在
里面填充 NaN。与前面介绍过的 pd.fillna() 函数类似，asfreq() 有一个 method 参数可以
设置填充缺失值的方式。下面将对工作日数据按天进行重新取样（即包含周末），结果如
图 3-7 所示：
```python
In[30]: fig, ax = plt.subplots(2, sharex=True)
 data = goog.iloc[:10]
 data.asfreq('D').plot(ax=ax[0], marker='o')
 data.asfreq('D', method='bfill').plot(ax=ax[1], style='-o')
 data.asfreq('D', method='ffill').plot(ax=ax[1], style='--o')
 ax[1].legend(["back-fill", "forward-fill"]);
```
图 3-7：asfreq() 向前填充与向后填充缺失值的结果对比
上面那幅图是原始数据：非工作日的股价是缺失值，所以不会出现在图上。而下面那幅图
通过向前填充与向后填充这两种方法填补了缺失值。
#### 2. 时间迁移
另一种常用的时间序列操作是对数据按时间进行迁移。Pandas 有两种解决这类问题的方
法：shift() 和 tshift()。简单来说，shift() 就是迁移数据，而 tshift() 就是迁移索引。
两种方法都是按照频率代码进行迁移。
下面我们将用 shift() 和 tshift() 这两种方法让数据迁移 900 天（如图 3-8 所示）：
```python
In[31]: fig, ax = plt.subplots(3, sharey=True)
 # 对数据应用时间频率，用向后填充解决缺失值
 goog = goog.asfreq('D', method='pad')
 goog.plot(ax=ax[0])
 goog.shift(900).plot(ax=ax[1])
 goog.tshift(900).plot(ax=ax[2])
 # 设置图例与标签
 local_max = pd.to_datetime('2007-11-05')
 offset = pd.Timedelta(900, 'D')
 ax[0].legend(['input'], loc=2)
 ax[0].get_xticklabels()[4].set(weight='heavy', color='red')
 ax[0].axvline(local_max, alpha=0.3, color='red')
 ax[1].legend(['shift(900)'], loc=2)
 ax[1].get_xticklabels()[4].set(weight='heavy', color='red')
 ax[1].axvline(local_max + offset, alpha=0.3, color='red')
 ax[2].legend(['tshift(900)'], loc=2) 

 ax[2].get_xticklabels()[1].set(weight='heavy', color='red')
 ax[2].axvline(local_max + offset, alpha=0.3, color='red');
 ```
图 3-8：对比 shift 与 tshift 方法
我们会发现，shift(900) 将数据向前推进了 900 天，这样图形中的一段就消失了（最左侧
就变成了缺失值），而 tshift(900) 方法是将时间索引值向前推进了 900 天。
这类迁移方法的常见使用场景就是计算数据在不同时段的差异。例如，我们可以用迁移后
的值来计算 Google 股票一年期的投资回报率（如图 3-9 所示）：
```python
In[32]: ROI = 100 * (goog.tshift(-365) / goog - 1)
 ROI.plot()
 plt.ylabel('% Return on Investment');
```
这可以帮助我们观察 Google 股票的总体特征：从图中可以看出，Google 的股票在 IPO 刚
刚成功之后最值得投资（图里的趋势很直观），在 2009 年年中开始衰退。
#### 3. 移动时间窗口
Pandas 处理时间序列数据的第 3 种操作是移动统计值（rolling statistics）。这些指标可以
通过 Series 和 DataFrame 的 rolling() 属性来实现，它会返回与 groupby 操作类似的结果
（详情请参见 3.9 节）。移动视图（rolling view）使得许多累计操作成为可能。
例如，可以通过下面的代码获取 Google 股票收盘价的一年期移动平均值和标准差（如图
3-10 所示）：
```python
In[33]: rolling = goog.rolling(365, center=True)
 data = pd.DataFrame({'input': goog,
 'one-year rolling_mean': rolling.mean(),
 'one-year rolling_std': rolling.std()})
 ax = data.plot(style=['-', '--', ':'])
 ax.lines[0].set_alpha(0.3)
 ```
图 3-10：Google 股票收盘价的移动统计值
与 groupby 操作一样，aggregate() 和 apply() 方法都可以用来自定义移动计算。
 
### 　更多学习资料
在这一节中，我们只是简单总结了 Pandas 时间序列工具的一些最常用功能，更详细的介
绍请参考 Pandas 在线文档“Time Series / Date”（http://pandas.pydata.org/pandas-docs/stable/
timeseries.html）节。
另一个优秀的资源是 Wes McKinney（Pandas 创建者）所著的《利用 Python 进行数据分
析》。虽然这本书已经有些年头了，但仍然是学习 Pandas 的好资源，尤其是这本书重点介
绍了时间序列工具在商业与金融业务中的应用，作者用大量笔墨介绍了工作日历、时区和
相关主题的具体内容。
你当然可以用 IPython 的帮助功能来浏览和深入探索上面介绍过的函数与方法，我个人认
为这是学习各种 Python 工具的最佳途径。
### 　案例：美国西雅图自行车统计数据的可视化
下面来介绍一个比较复杂的时间序列数据，统计自 2012 年以来每天经过美国西雅图弗
莱蒙特桥（http://www.openstreetmap.org/#map=17/47.64813/-122.34965）上的自行车的数
量，数据由安装在桥东西两侧人行道的传感器采集。小时统计数据可以在 http://data.seattle.
gov/ 下载，还有一个数据集的直接下载链接 https://data.seattle.gov/Transportation/FremontBridge-Hourly-Bicycle-Counts-by-Month-Octo/65db-xm6k。
截至 2016 年夏，CSV 数据可以用以下命令下载：
```python
In[34]:
# !curl -o FremontBridge.csv
# https://data.seattle.gov/api/views/65db-xm6k/rows.csv?accessType=DOWNLOAD
```
下好数据之后，可以用 Pandas 读取 CSV 文件获取一个 DataFrame。我们将 Date 作为时间
索引，并希望这些日期可以被自动解析：
```python
In[35]:
data = pd.read_csv('FremontBridge.csv', index_col='Date', parse_dates=True)
data.head()
Out[35]: Fremont Bridge West Sidewalk \\
 Date
 2012-10-03 00:00:00 4.0
 2012-10-03 01:00:00 4.0
 2012-10-03 02:00:00 1.0
 2012-10-03 03:00:00 2.0
 2012-10-03 04:00:00 6.0
 Fremont Bridge East Sidewalk
 Date
 2012-10-03 00:00:00 9.0
 2012-10-03 01:00:00 6.0
 2012-10-03 02:00:00 1.0
 2012-10-03 03:00:00 3.0
 2012-10-03 04:00:00 1.0
 ```
为了方便后面的计算，缩短数据集的列名，并新增一个 Total 列：
```python
In[36]: data.columns = ['West', 'East']
 data['Total'] = data.eval('West + East')
 ```
现在来看看这三列的统计值：
```python
In[37]: data.dropna().describe()
Out[37]: West East Total 

 count 33544.000000 33544.000000 33544.000000
 mean 61.726568 53.541706 115.268275
 std 83.210813 76.380678 144.773983
 min 0.000000 0.000000 0.000000
 25% 8.000000 7.000000 16.000000
 50% 33.000000 28.000000 64.000000
 75% 80.000000 66.000000 151.000000
 max 825.000000 717.000000 1186.000000
 ```
#### 1. 数据可视化
通过可视化，我们可以对数据集有一些直观的认识。先为原始数据画图（如图 3-11
所示）：
```python
In[38]: %matplotlib inline
 import seaborn; seaborn.set()
In[39]: data.plot()
 plt.ylabel('Hourly Bicycle Count');
 ```
图 3-11：弗莱蒙特桥每小时通行的自行车数量
在图中显示大约 25 000 小时的样本数据对我们来说实在太多了，因此可以通过重新取样将
数据转换成更大的颗粒度，比如按周累计（如图 3-12 所示）：
```python
In[40]: weekly = data.resample('W').sum()
 weekly.plot(style=[':', '--', '-'])
 plt.ylabel('Weekly bicycle count');
 ```
这就显示出一些季节性的特征了。正如你所想，夏天骑自行车的人比冬天多，而且某个季
节中每一周的自行车数量也在变化（可能与天气有关，详情请参见 5.6 节）。
图 3-12：弗莱蒙特桥每周通行的自行车数量
另一种对数据进行累计的简便方法是用 pd.rolling_mean()4 函数求移动平均值。下面将计
算数据的 30 日移动均值，并让图形在窗口居中显示（center=True）（如图 3-13 所示）：
```python
In[41]: daily = data.resample('D').sum()
 daily.rolling(30, center=True).mean().plot(style=[':', '--', '-'])
 plt.ylabel('mean of 30 days count');
 ```
图 3-13：每 30 日自行车的移动日均值
由于窗口太小，现在的图形还不太平滑。我们可以用另一个移动均值的方法获得更平滑的
注 4：原书代码与正文不符。作者在正文中说“用 pd.rolling_meaning() 函数”，但作者代码中 daily.rolling
(30,center=True).sum() 等价于 pd.rolling_sum()。另外，Pandas 文档提到，pd.rolling_mean 方法即
将被废弃，用 DataFrame.rolling(center=False,window=D).mean() 的形式代替 pd.rolling_mean()。考虑
到原文图题是“30 天自行车数量”，因此按照 30 天的日均值作相应的修改。——译者注
图形，例如高斯分布时间窗口。下面的代码（可视化后如图 3-14 所示）将设置窗口的宽度
（选择 50 天）和窗口内高斯平滑的宽度（选择 10 天）：
```python
In[42]:
daily.rolling(50, center=True,
 win_type='gaussian').sum(std=10).plot(style=[':', '--', '-']);
 ```
图 3-14：用高斯平滑方法处理每周自行车的移动均值
### 2. 深入挖掘数据
虽然我们已经从图 3-14 的平滑数据图观察到了数据的总体趋势，但是它们还隐藏了一些有
趣的特征。例如，我们可能希望观察单日内的小时均值流量，这可以通过 GroupBy（详情
请参见 3.9 节）操作来解决（如图 3-15 所示）：
```python
In[43]: by_time = data.groupby(data.index.time).mean()
 hourly_ticks = 4 * 60 * 60 * np.arange(6)
 by_time.plot(xticks=hourly_ticks, style=[':', '--', '-']);
```
小时均值流量呈现出十分明显的双峰分布特征，早间峰值在上午 8 点，晚间峰值在下午 5 点。
这充分反映了过桥上下班往返自行车流量的特征。进一步分析会发现，桥西的高峰在早上（因
为人们每天会到西雅图的市中心上班），而桥东的高峰在下午（下班再从市中心离开）。
我们可能还会对周内每天的变化产生兴趣，这时依然可以通过一个简单的 groupby 来实现
（如图 3-16 所示）：
```python
In[44]: by_weekday = data.groupby(data.index.dayofweek).mean()
 by_weekday.index = ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun']
 by_weekday.plot(style=[':', '--', '-']);
 ```
图 3-16：每周每天的自行车流量
工作日与周末的自行车流量差十分显著，周一到周五通过的自行车差不多是周六、周日的
两倍。
看到这个特征之后，让我们用一个复合 groupby 来观察一周内工作日与双休日每小时的数
据。用一个标签表示双休日和工作日的不同小时：
```python
In[45]: weekend = np.where(data.index.weekday < 5, 'Weekday', 'Weekend')
 by_time = data.groupby([weekend, data.index.time]).mean()
 ```
现在用一些 Matplotlib 工具（详情请参见 4.10 节）画出两张图（如图 3-17 所示）：
```python
In[46]: import matplotlib.pyplot as plt
 fig, ax = plt.subplots(1, 2, figsize=(14, 5))
 by_time.ix['Weekday'].plot(ax=ax[0], title='Weekdays',
 xticks=hourly_ticks, style=[':', '--', '-'])
 by_time.ix['Weekend'].plot(ax=ax[1], title='Weekends',
 xticks=hourly_ticks, style=[':', '--', '-']);
```
图 3-17：工作日与双休日每小时的自行车流量
结果很有意思，我们会发现工作日的自行车流量呈双峰通勤模式（bimodal commute pattern），
而到了周末就变成了单峰娱乐模式（unimodal recreational pattern）。假如继续挖掘数据应该还
会发现更多有趣的信息，比如研究天气、温度、一年中的不同时间以及其他因素对人们通勤
模式的影响。关于更深入的分析内容，请参考我的博文“Is Seattle Really Seeing an Uptick In
Cycling?”（https://jakevdp.github.io/blog/2014/06/10/is-seattle-really-seeing-an-uptick-in-cycling/），
里面用数据的子集作了一些分析。我们将在 5.6 节继续使用这个数据集。
 
 
