# day46 深入研究Numpy2
## 聚合：最小值、最大值和其他值
### 1.数组值求和   
设想计算一个数组中所有元素的和。Python 本身可用内置的 sum 函数 来实现：
```python
In[1]: import numpy as np 
 
In[2]: L = np.random.random(100)       
sum(L) 
 
Out[2]: 55.61209116604941
```
它的语法和 NumPy 的 sum 函数非常相似，并且在这个简单的例子中的结果也是一样的：
```python
In[3]: np.sum(L) 
 
Out[3]: 55.612091166049424
```
但是，因为 NumPy 的 sum 函数在编译码中执行操作，所以 NumPy 的操作计算得更 快一些：
```python
In[4]: big_array = np.random.rand(1000000)        
%timeit sum(big_array)        
%timeit np.sum(big_array) 

10 loops, best of 3: 104 ms per loop 
1000 loops, best of 3: 442 µs per loop
```
但是需要注意，sum 函数和 np.sum 函数并不等同，这有时会导致混淆。
尤其是它们各自 的可选参数都有不同的含义，np.sum 函数是知道数组的维度的，这一点将在接下来的部 分讲解。

### 2.最小值和最大值 
Python 也有内置的 min 函数和 max 函数，分别被用于获取给定数组的最小值和最 大值：
```python
In[5]: min(big_array), max(big_array) 
 
Out[5]: (1.1717128136634614e-06, 0.9999976784968716)
```
NumPy 对应的函数也有类似的语法，并且也执行得更快：
```python
In[6]: np.min(big_array), np.max(big_array) 
 
Out[6]: (1.1717128136634614e-06, 0.9999976784968716) 
 
In[7]: %timeit min(big_array)        
%timeit np.min(big_array) 
 
10 loops, best of 3: 82.3 ms per loop 
1000 loops, best of 3: 497 µs per loop
```
对于 min、max、sum 和其他 NumPy 聚合，一种更简洁的语法形式是数组对象直接调用这些 方法：
```python
In[8]: print(big_array.min(), big_array.max(), big_array.sum()) 
 
1.17171281366e-06 0.999997678497 499911.628197
```
当你操作 NumPy 数组时，确保你执行的是 NumPy 版本的聚合。 
1. 多维度聚合 
一种常用的聚合操作是沿着一行或一列聚合。例如，假设你有一些数据存储在二维数组中：
```python
In[9]: M = np.random.random((3, 4))        
print(M) 
 
[[ 0.8967576   0.03783739  0.75952519  0.06682827]
[ 0.8354065   0.99196818  0.19544769  0.43447084] 
[ 0.66859307  0.15038721  0.37911423  0.6687194]]
```
默认情况下，每一个 NumPy 聚合函数将会返回对整个数组的聚合结果：
```python
In[10]: M.sum()
 
Out[10]: 6.0850555667307118
```
聚合函数还有一个参数，用于指定沿着哪个轴的方向进行聚合。
例如，可以通过指定 axis=0 找到每一列的最小值：
```python
In[11]: M.min(axis=0) 
 
Out[11]: array([ 0.66859307,  0.03783739,  0.19544769,  0.06682827])
```
这个函数返回四个值，对应四列数字的计算值。
同样，也可以找到每一行的最大值：
```python
In[12]: M.max(axis=1) 
 
Out[12]: array([ 0.8967576 ,  0.99196818,  0.6687194])
```
其他语言的用户会对轴的指定方式比较困惑。axis 关键字指定的是数组将会被折叠的维 度，而不是将要返回的维度。因此指定 axis=0 意味着第一个轴将要被折叠——对于二维数 组，这意味着每一列的值都将被聚合。

### 3.示例：美国总统的身高是多少 
这里举一个简单的例子——计算所有美 国总统的身高。这个数据在 president_heights.csv 文件中，
是一个简单的用逗号分隔的标签 和值的列表：
```python
In[13]: !head -4 data/president_heights.csv 
 
order,name,height(cm) 
1,George Washington,189 
2,John Adams,170 
3,Thomas Jefferson,189
```
我们将用 Pandas 包来读文件并抽取身高信息。（请注意，身高的计量单位是厘米。）第 3 章 将更全面地介绍 Pandas：
```python
In[14]: import pandas as pd         
data = pd.read_csv('data/president_heights.csv')         
heights = np.array(data['height(cm)'])         
print(heights) 
 
[189 170 189 163 183 171 185 168 173 183 173 173 
175 178 183 193 178 173  174 183 183 168 170 178 
182 180 183 178 182 188 175 179 183 193 182 183  
177 185 188 188 182 185]
```
有了这个数据数组后，就可以计算很多概括统计值了：
```python
In[15]: print("Mean height:       ", heights.mean()) 
print("Standard deviation:", heights.std())        
print("Minimum height:    ", heights.min())        
print("Maximum height:    ", heights.max()) 
 
Mean height:        179.738095238 
Standard deviation: 6.93184344275 
Minimum height:     163 
Maximum height:     193
```
请注意，在这个例子中，聚合操作将整个数组缩减到单个概括值，这个概括值给出了这些 数值的分布信息。我们也可以计算分位数：
```python
In[16]: print("25th percentile:   ", np.percentile(heights, 25))
print("Median:            ", np.median(heights))
print("75th percentile:   ", np.percentile(heights, 75)) 
 
25th percentile:    174.25 
Median:             182.0 
75th percentile:    183.0
```
可以看到，美国总统的身高中位数是 182cm，或者说不到 6 英尺。

## 数组的计算：广播
通过通用函数的向量化操作来减少缓慢的 Python 循环， 另外一种向量化操作的方法是利用 NumPy 的广播功能。广播可以简单理解为用于不同大 小数组的二进制通用函数（加、减、乘等）的一组规则。

### 1.广播的介绍 
对于同样大小的数组，二进制操作是对相应元素逐个计算：
```python
In[1]: import numpy as np 
 
In[2]: a = np.array([0, 1, 2])
b = np.array([5, 5, 5])        
a + b 
 
Out[2]: array([5, 6, 7])
```
广播允许这些二进制操作可以用于不同大小的数组。例如，可以简单地将一个标量
（可以 认为是一个零维的数组）和一个数组相加：
```python
In[3]: a + 5 
 
Out[3]: array([5, 6, 7])
```
我们可以认为这个操作是将数值 5 扩展或重复至数组 [5, 5, 5]，然后执行加法。
NumPy 广播功能的好处是，这种对值的重复实际上并没有发生，但是这是一种很好用的理解广播 的模型。
我们同样也可以将这个原理扩展到更高维度的数组。观察以下将一个一维数组和一个二维 数组相加的结果：
```python
In[4]: M = np.ones((3, 3))       
M 
 
Out[4]: array([[ 1.,  1.,  1.],               
[ 1.,  1.,  1.],               
[ 1.,  1.,  1.]]) 
 
In[5]: M + a 
 
Out[5]: array([[ 1.,  2.,  3.],               
[ 1.,  2.,  3.],                
[ 1.,  2.,  3.]])
```
这里这个一维数组就被扩展或者广播了。它沿着第二个维度扩展，扩展到匹配 M 数组 的形状。
以上的这些例子理解起来都相对容易，更复杂的情况会涉及对两个数组的同时广播，例如 以下示例：

```python
In[6]: a = np.arange(3)       
b = np.arange(3)[:, np.newaxis] 
 
       print(a)
       print(b) 
 
[0 1 2] 
[[0]  
[1]  
[2]] 
 
In[7]: a + b 
 
Out[7]: array([[0, 1, 2],
[1, 2, 3],               
[2, 3, 4]])
```
正如此前将一个值扩展或广播以匹配另外一个数组的形状，这里将 a 和 b 都进行了扩展来 
匹配一个公共的形状，最终的结果是一个二维数组。
### 2.广播的规则
NumPy 的广播遵循一组严格的规则，设定这组规则是为了决定两个数组间的操作。
• 规则 1：如果两个数组的维度数不相同，那么小维度数组的形状将会在最左边补 1。
• 规则 2：如果两个数组的形状在任何一个维度上都不匹配，那么数组的形状会沿着维度 
为 1 的维度扩展以匹配另外一个数组的形状。 
• 规则 3：如果两个数组的形状在任何一个维度上都不匹配并且没有任何一个维度等于 1， 那么会引发异常。
为了更清楚地理解这些规则，来看几个具体示例。
1. 广播示例1 
将一个二维数组与一个一维数组相加：
```python
In[8]: M = np.ones((2, 3))        
a = np.arange(3)
```
来看这两个数组的加法操作。两个数组的形状如下：
```python
M.shape = (2, 3)
a.shape = (3,)
```
可以看到，根据规则 1，数组 a 的维度数更小，所以在其左边补 1：
```python
M.shape -> (2, 3) 
a.shape -> (1, 3)
```
根据规则 2，第一个维度不匹配，因此扩展这个维度以匹配数组：
```python
M.shape -> (2, 3) 
a.shape -> (2, 3)
```
现在两个数组的形状匹配了，可以看到它们的最终形状都为 (2, 3)：
```python
In[9]: M + a 
 
Out[9]: array([[ 1.,  2.,  3.], 
[ 1.,  2.,  3.]])
```
2. 广播示例2 
来看两个数组均需要广播的示例：
```python
In[10]: a = np.arange(3).reshape((3, 1))        
b = np.arange(3)
```
同样，首先写出两个数组的形状：
```python
a.shape = (3, 1) 
b.shape = (3,)
```
规则 1 告诉我们，需要用 1 将 b 的形状补全：
```python
a.shape -> (3, 1) 
b.shape -> (1, 3)
```
规则 2 告诉我们，需要更新这两个数组的维度来相互匹配：
```python
a.shape -> (3, 3)
b.shape -> (3, 3)
```
因为结果匹配，所以这两个形状是兼容的，可以看到以下结果：
```python
In[11]: a + b 
 
Out[11]: array([[0, 1, 2], 
[1, 2, 3],                
[2, 3, 4]])
```
3. 广播示例3 
现在来看一个两个数组不兼容的示例：
```python
In[12]: M = np.ones((3, 2))        
a = np.arange(3)
```
和第一个示例相比，这里有个微小的不同之处：矩阵 M 是转置的。那么这将如何影响计算 
呢？两个数组的形状如下：
```python
M.shape = (3, 2)
a.shape = (3,)
```
同样，规则 1 告诉我们，a 数组的形状必须用 1 进行补全：
```python
M.shape -> (3, 2) 
a.shape -> (1, 3)
```
根据规则 2，a 数组的第一个维度进行扩展以匹配 M 的维度：
```python
M.shape -> (3, 2) 
a.shape -> (3, 3)
```
现在需要用到规则 3——最终的形状还是不匹配，因此这两个数组是不兼容的。当我们执 
行运算时会看到以下结果：
```python
In[13]: M + a 
 
--------------------------------------------------------------------------- 
 
ValueError                                Traceback (most recent call last) 
 
<ipython-input-13-9e16e9f98da6> in <module>() ----> 1 M + a 
 
 
ValueError: operands could not be broadcast together with shapes (3,2) (3,)
```
请注意，这里可能发生的混淆在于：你可能想通过在 a 数组的右边补 1，而不是左边补 1，
让 a 和 M 的维度变得兼容。但是这不被广播的规则所允许。这种灵活性在有些情景中可能 会有用，
但是它可能会导致结果模糊。如果你希望实现右边补全，可以通过变形数组来实
现（将会用到 np.newaxis 关键字，详情请参见 2.2 节）：
```python
In[14]: a[:, np.newaxis].shape 
 
Out[14]: (3, 1) 
 
In[15]: M + a[:, np.newaxis] 
 
Out[15]: array([[ 1.,  1.],  
[ 2.,  2.],              
[ 3.,  3.]])
```
另外也需要注意，这里仅用到了 + 运算符，而这些广播规则对于任意二进制通用函数都是 
适用的。例如这里的 logaddexp(a, b) 函数，比起简单的方法，该函数计算 log(exp(a) + exp(b)) 
更准确：
```python
In[16]: np.logaddexp(M, a[:, np.newaxis]) 
 
Out[16]: array([[ 1.31326169,  1.31326169],                 
[ 1.69314718,  1.69314718],                
[ 2.31326169,  2.31326169]])
```
### 3.　广播的实际应用 
1. 数组的归一化 
在前面的一节中，我们看到通用函数让 NumPy 用户免于写很慢的Python 循环。广播进 
一步扩展了这个功能，一个常见的例子就是数组数据的归一化。假设你有一个有 10 个观
察值的数组，每个观察值包含 3 个数值。按照惯例（详情请参见 5.2 节），我们将用一个 10×3 
的数组存放该数据：
```python
In[17]: X = np.random.random((10, 3))
```
我们可以计算每个特征的均值，计算方法是利用 mean 函数沿着第一个维度聚合：
```python
In[18]: Xmean = X.mean(0)         
Xmean 
 
Out[18]: array([ 0.53514715,  0.66567217,  0.44385899])
```
现在通过从 X 数组的元素中减去这个均值实现归一化（该操作是一个广播操作）：
```python
In[19]: X_centered = X - Xmean
```
为了进一步核对我们的处理是否正确，可以查看归一化的数组的均值是否接近 0：
```python
In[20]: X_centered.mean(0) 
 
Out[20]: array([  2.22044605e-17,  -7.77156117e-17,  -1.66533454e-17])
```
在机器精度范围内，该均值为 0。
2. 画一个二维函数 
广播另外一个非常有用的地方在于，它能基于二维函数显示图像。我们希望定义一个函数
z = f (x, y)，可以用广播沿着数值区间计算该函数：
```python
In[21]: # x和y表示0~5区间50个步长的序列        
x = np.linspace(0, 5, 50)        
y = np.linspace(0, 5, 50)[:, np.newaxis] 

z = np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)
```
我们将用 Matplotlib 来画出这个二维数组（这些工具将在 4.6 节中详细介绍）：
```python
In[22]: %matplotlib inline        
import matplotlib.pyplot as plt 
 
In[23]: plt.imshow(z, origin='lower', extent=[0, 5, 0, 5], 
cmap='viridis')        
plt.colorbar();
```
## 比较、掩码和布尔逻辑
当你想基于某些准则 来抽取、修改、计数或对一个数组中的值进行其他操作时，
掩码就可以派上用场了。例如 你可能希望统计数组中有多少值大于某一个给定值，
或者删除所有超出某些门限值的异常 点。在 NumPy 中，布尔掩码通常是完成这类任务的最高效方式。

### 1.　示例：统计下雨天数 
### 2. 和通用函数类似的比较操作 
我们看到用 +、-、*、/ 和其他一些 运算符实现了数组的逐元素操作。NumPy 
还实现了如 <（小于）和 >（大于）的逐元素比 较的通用函数。这
些比较运算的结果是一个布尔数据类型的数组。一共有 6 种标准的比较 操作：
```python
In[4]: x = np.array([1, 2, 3, 4, 5]) 
 
In[5]: x < 3  # 小于 
 
Out[5]: array([ True,  True, False, False, False], dtype=bool) 
 
In[6]: x > 3  # 大于 
 

Out[6]: array([False, False, False,  True,  True], dtype=bool) 
 
In[7]: x <= 3  # 小于等于 
 
Out[7]: array([ True,  True,  True, False, False], dtype=bool) 
 
In[8]: x >= 3  # 大于等于 
 
Out[8]: array([False, False,  True,  True,  True], dtype=bool) 
 
In[9]: x != 3  # 不等于 
 
Out[9]: array([ True,  True, False,  True,  True], dtype=bool) 
 
In[10]: x == 3  # 等于 
 
Out[10]: array([False, False,  True, False, False], dtype=bool)
```
另外，利用复合表达式实现对两个数组的逐元素比较也是可行的：
```python
In[11]: (2 * x) == (x ** 2) 
 
Out[11]: array([False,  True, False, False, False], dtype=bool)
```
和算术运算符一样，比较运算操作在 NumPy 中也是借助通用函数来实现的。例如当你写 x < 3 时，NumPy 内部会使用 np.less(x, 3)。
和算术运算通用函数一样，这些比较运算通用函数也可以用于任意形状、大小的数组。下 面是一个二维数组的示例：
```python
In[12]: rng = np.random.RandomState(0)       
x = rng.randint(10, size=(3, 4))        
x 
 
Out[12]: array([[5, 0, 3, 3],              
[7, 9, 3, 5],                
[2, 4, 7, 6]]) 
 
In[13]: x < 6 
 
Out[13]: array([[ True,  True,  True,  True],             
[False, False,  True,  True], 
[ True,  True, False, False]], dtype=bool)
```
这样每次计算的结果都是布尔数组了。NumPy 提供了一些简明的模式来操作这些布尔 结果。
###3.　操作布尔数组 
给定一个布尔数组，你可以实现很多有用的操作。首先打印出此前生成的二维数组 x：
```python
In[14]: print(x) 
 
[[5 0 3 3] 
[7 9 3 5] 
[2 4 7 6]]
```
1. 统计记录的个数 
如果需要统计布尔数组中 True 记录的个数，可以使用 np.count_nonzero 函数：
```python
In[15]: # 有多少值小于6？        
np.count_nonzero(x < 6) 
 
Out[15]: 8
```
我们看到有 8 个数组记录是小于 6 的。另外一种实现方式是利用 np.sum。
在这个例子中， False 会被解释成 0，True 会被解释成 1：
```python
In[16]: np.sum(x < 6) 
 
Out[16]: 8
```
sum() 的好处是，和其他 NumPy 聚合函数一样，这个求和也可以沿着行或列进行：
```python
In[17]: # 每行有多少值小于6？         
np.sum(x < 6, axis=1) 
 
Out[17]: array([4, 2, 2])
```
这是矩阵中每一行小于 6 的个数。
如要快速检查任意或者所有这些值是否为True，可以用（你一定猜到了）np.any() 或 np.all()：
```python
In[18]: # 有没有值大于8？        
np.any(x > 8) 
 
Out[18]: True 
 
In[19]: # 有没有值小于0？       
np.any(x < 0) 
 
Out[19]: False 
 
In[20]: # 是否所有值都小于10？       
np.all(x < 10) 

Out[20]: True 
 
In[21]: # 是否所有值都等于6？        
np.all(x == 6) 
 
Out[21]: False
```
np.all() 和 np.any() 也可以用于沿着特定的坐标轴，例如：
```python
In[22]: # 是否每行的所有值都小于8？        
np.all(x < 8, axis=1) 
 
Out[22]: array([ True, False,  True], dtype=bool)
```
这里第 1 行和第 3 行的所有元素都小于 8，而第 2 行不是所有元素都小于 8。
最后需要提醒的是，正如在 2.4 节中提到的，Python 有内置的 sum()、any() 和 all() 函数，
这些函数在 NumPy 中有不同的语法版本。如果在多维数组上混用这两个版本，会导致失 
败或产生不可预知的错误结果。因此，确保在以上的示例中用的都是 np.sum()、np.any() 
和 np.all() 函数。
2. 布尔运算符 
我们已经看到该如何统计所有降水量小于 4 英寸或者大于 2 英寸的天数，
但是如果我们想 统计降水量小于 4 英寸且大于 2 英寸的天数该如何操作呢？这可以通过 
Python 的逐位逻辑 运算符（bitwise logic operator）&、|、^ 和 ~ 来实现。同标准的算术运算符一样，
NumPy 用通用函数重载了这些逻辑运算符，这样可以实现数组的逐位运算（通常是布尔运算）。
例如，可以写如下的复合表达式：
```python
In[23]: np.sum((inches > 0.5) & (inches < 1)) 
 
Out[23]: 29
```
可以看到，降水量在 0.5 英寸 ~1 英寸间的天数是 29 天。
请注意，这些括号是非常重要的，因为有运算优先级规则。如果去掉这些括号，该表达式
会变成以下形式，这会导致运行错误：
```python
inches > (0.5 & inches) < 1
```
利用 A AND B 和 NOT (A OR B) 的等价原理（你应该在基础逻辑课程中学习过），可以用
另外一种形式实现同样的结果：
```python
In[24]: np.sum(~( (inches <= 0.5) | (inches >= 1) )) 
 
Out[24]: 29
```
将比较运算符和布尔运算符合并起来用在数组上，可以实现更多有效的逻辑运算操作。
以下表格总结了逐位的布尔运算符和其对应的通用函数。

运算符 对应通用函数 
& np.bitwise_and 
| np.bitwise_or 
^ np.bitwise_xor
~ np.bitwise_not
利用这些工具，就可以回答那些关于天气数据的问题了。以下的示例是结合使用掩码和聚 合实现的结果计算：
```python
In[25]: print("Number days without rain:      ", np.sum(inches == 0))
print("Number days with rain:         ", np.sum(inches != 0)) 
print("Days with more than 0.5 inches:", np.sum(inches > 0.5))
print("Rainy days with < 0.1 inches  :", np.sum((inches > 0) &                                                         (inches < 0.2))) 
 
Number days without rain:       215
Number days with rain:          150
Days with more than 0.5 inches: 37
Rainy days with < 0.1 inches  : 75
```
### 4.　将布尔数组作为掩码
在前面的小节中，我们看到了如何直接对布尔数组进行聚合计算。一种更强大的模式是使 
用布尔数组作为掩码，通过该掩码选择数据的子数据集。以前面小节用过的 x 数组为例，
假设我们希望抽取出数组中所有小于 5 的元素：
```python
In[26]: x 
 
Out[26]: array([[5, 0, 3, 3],              
[7, 9, 3, 5],                
[2, 4, 7, 6]])
```
如前面介绍过的方法，利用比较运算符可以得到一个布尔数组：
```python
In[27]: x < 5 
 
Out[27]: array([[False,  True,  True,  True],          
[False, False,  True, False],               
[ True,  True, False, False]], dtype=bool)
```
现在为了将这些值从数组中选出，可以进行简单的索引，即掩码操作：
```python
In[28]: x[x < 5] 
 
Out[28]: array([0, 3, 3, 3, 2, 4])
```
现在返回的是一个一维数组，它包含了所有满足条件的值。换句话说，所有的这些值是掩 码数组对应位置为 True 的值。
现在，可以对这些值做任意操作，例如可以根据西雅图降水数据进行一些相关统计：
```python
In[29]: # 为所有下雨天创建一个掩码 rainy = (inches > 0) 
 
# 构建一个包含整个夏季日期的掩码（6月21日是第172天）
```python
summer = (np.arange(365) - 172 < 90) & (np.arange(365) - 172 > 0) 
 
print("Median precip on rainy days in 2014 (inches):   ",       np.median(inches[rainy])) 
print("Median precip on summer days in 2014 (inches):  ",  np.median(inches[summer])) 
print("Maximum precip on summer days in 2014 (inches): ",  np.max(inches[summer])) 
print("Median precip on non-summer rainy days (inches):",    np.median(inches[rainy & ~summer])) 
 
Median precip on rainy days in 2014 (inches):0.194881889764 
Median precip on summer days in 2014 (inches):  0.0 
Maximum precip on summer days in 2014 (inches): 0.850393700787 
Median precip on non-summer rainy days (inches): 0.200787401575
```
通过将布尔操作、掩码操作和聚合结合，可以快速回答对数据集提出的这类问题。、

使用关键字 and/or 与使用逻辑操作运算符 &/|
人们经常困惑于关键字 and 和 or，以及逻辑操作运算符 & 和 | 的区别是什么，什么时 候该选择哪一种？
它们的区别是：and 和 or 判断整个对象是真或假，而 & 和 | 是指每个对象中的比特位。
当你使用 and 或 or 时，就等于让 Python 将这个对象当作整个布尔实体。在 Python 中， 所有非零的整数都会被当作是 True：
```python
In[30]: bool(42), bool(0) 
 
Out[30]: (True, False) 
 
In[31]: bool(42 and 0) 
 
Out[31]: False 
 
In[32]: bool(42 or 0) 
 
Out[32]: True
```
当你对整数使用 & 和 | 时，表达式操作的是元素的比特，将 and 或 or 应用于组成该数 字的每个比特：
```python
In[33]: bin(42) 
 
Out[33]: '0b101010' 
 
In[34]: bin(59) 
 
Out[34]: '0b111011'

In[35]: bin(42 & 59) 
 
Out[35]: '0b101010' 
 
In[36]: bin(42 | 59) 
 
Out[36]: '0b111011'
```
请注意，& 和 | 运算时，对应的二进制比特位进行比较以得到最终结果。
当你在 NumPy 中有一个布尔数组时，该数组可以被当作是由比特字符组成的，其中 1 = True、0 = False。这样的数组可以用上面介绍的方式进行 & 和 | 的操作：
```python
In[37]: A = np.array([1, 0, 1, 0, 1, 0], dtype=bool)      
B = np.array([1, 1, 1, 0, 1, 1], dtype=bool)      
A | B 
 
Out[37]: array([ True,  True,  True, False,  True,  True], dtype=bool)
```
而用 or 来计算这两个数组时，Python 会计算整个数组对象的真或假，这会导致程序出错：
```python
In[38]: A or B 
 
--------------------------------------------------------------------------- 
 
ValueError                                Traceback (most recent call last) 
 
<ipython-input-38-5d8e4f2e21c0> in <module>() ----> 1 A or B 
 
 
ValueError: The truth value of an array with more than one element is...
```
同样，对给定数组进行逻辑运算时，你也应该使用 | 或 &，而不是 or 或 and：
```python
In[39]: x = np.arange(10)         
(x > 4) & (x < 8) 
 
Out[39]: array([False, False, ...,  True,  True, False, False], dtype=bool)
```
如果试图计算整个数组的真或假，程序也同样会给出 ValueError 的错误：
```python
In[40]: (x > 4) and (x < 8) 
 
--------------------------------------------------------------------------- 
 
ValueError                                Traceback (most recent call last) 
 
<ipython-input-40-3d24f1ffd63d> in <module>() ----> 1 (x > 4) and (x < 8) 
 
ValueError: The truth value of an array with more than one element is...

因此可以记住：and 和 or 对整个对象执行单个布尔运算，而 & 和 | 对一个对象的内 
容（单个比特或字节）执行多个布尔运算。对于 NumPy 布尔数组，后者是常用的
```



