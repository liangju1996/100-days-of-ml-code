## 高性能Pandas： eval()与query()Python 

数据科学生态环境的强大力量建立在 NumPy 与 Pandas 的基础之上，并通过直观的语法将基本操作转换成 C 语言：在 NumPy 里是向量化 / 广播运
算，在 Pandas 里是分组型的运算。虽然这些抽象功能可以简洁高效地解决许多问题，但是它们经常需要创建临时中间对象，这样就会占用大量的计算时间与内存。

### query()与eval()的设计动机： 复合代数式

前面已经介绍过， NumPy 与 Pandas 都支持快速的向量化运算。例如，你可以对下面两个数组进行求和：
```python
In[1]: import numpy as np
rng = np.random.RandomState(42)
x = rng.rand(1E6)
y = rng.rand(1E6)
%timeit x + y
100 loops, best of 3: 3.39 ms per loop
```
但是这种运算在处理复合代数式（compound expression）问题时的效率比较低，每段中间过程都需要显式地分配内存。如果 x 数组和 y 数组非常大，这么运算就会占用大量的时间和内存消耗。 Numexpr 程序库可以让你在不为中间过程分配全部内存的前提下，完成元素到元素的复合代数式运算。
每段中间过程都需要显式地分配内存。如果 x 数组和 y 数组非常大，这么运算就会占用大量的时间和内存消耗。 Numexpr 程序库可以让你在不为中间过程分配全部内存的前提下，完成元素到元素的复合代数式运算。
```python
In[6]: import pandas as pd
nrows, ncols = 100000, 100
rng = np.random.RandomState(42)
df1, df2, df3, df4 = (pd.DataFrame(rng.rand(nrows, ncols))
for i in range(4))
```
如果要用普通的 Pandas 方法计算四个 DataFrame 的和，可以这么写：
```python
In[7]: %timeit df1 + df2 + df3 + df4
10 loops, best of 3: 87.1 ms per loop
```
也可以通过 pd.eval 和字符串代数式计算并得出相同的结果：
```python
In[8]: %timeit pd.eval('df1 + df2 + df3 + df4')
10 loops, best of 3: 42.2 ms per loop
```
这个 eval() 版本的代数式比普通方法快一倍（而且内存消耗更少），结果也是一样的：
```python
In[9]: np.allclose(df1 + df2 + df3 + df4,
pd.eval('df1 + df2 + df3 + df4'))
Out[9]: True
```
(1) 算术运算符。 pd.eval() 支持所有的算术运算符，例如：
```python
In[11]: result1 = -df1 * df2 / (df3 + df4) - df5
result2 = pd.eval('-df1 * df2 / (df3 + df4) - df5')
np.allclose(result1, result2)
Out[11]: True
```
(2) 比较运算符。 pd.eval() 支持所有的比较运算符，包括链式代数式（chained expression）：
```python
In[12]: result1 = (df1 < df2) & (df2 <= df3) & (df3 != df4)
result2 = pd.eval('df1 < df2 <= df3 != df4')
np.allclose(result1, result2)
Out[12]: True
```
(3) 位运算符。 pd.eval() 支持 &（与）和 |（或）等位运算符：
```python
In[13]: result1 = (df1 < 0.5) & (df2 < 0.5) | (df3 < df4)
result2 = pd.eval('(df1 < 0.5) & (df2 < 0.5) | (df3 < df4)')
np.allclose(result1, result2)
Out[13]: True
```
(4) 对象属性与索引。 pd.eval() 可以通过 obj.attr 语法获取对象属性，通过 obj[index] 语
法获取对象索引：
```python
In[15]: result1 = df2.T[0] + df3.iloc[1]
result2 = pd.eval('df2.T[0] + df3.iloc[1]')
np.allclose(result1, result2)
Out[15]: True
```
(5) 其他运算。目前 pd.eval() 还不支持函数调用、条件语句、循环以及更复杂的运算。如果你想要进行这些运算，可以借助 Numexpr 来实现。
### 用DataFrame.eval()实现列间运算

由于 pd.eval() 是 Pandas 的顶层函数，因此 DataFrame 有一个 eval() 方法可以做类似的运算。使用 eval() 方法的好处是可以借助列名称进行运算，示例如下：
```python
In[16]: df = pd.DataFrame(rng.rand(1000, 3), columns=['A', 'B', 'C'])
        df.head()
Out[16]: A B C
0 0.375506 0.406939 0.069938
1 0.069087 0.235615 0.154374
2 0.677945 0.433839 0.652324
3 0.264038 0.808055 0.347197
4 0.589161 0.252418 0.557789
1. 用DataFrame.eval()新增列
```
除了前面介绍的运算功能， DataFrame.eval() 还可以创建新的列。还用前面的 DataFrame
来演示，列名是 'A'、 'B' 和 'C':
```python
In[19]: df.head()
Out[19]: A B C
0 0.375506 0.406939 0.069938
1 0.069087 0.235615 0.154374
2 0.677945 0.433839 0.652324
3 0.264038 0.808055 0.347197
4 0.589161 0.252418 0.557789
```
@ 符号表示“这是一个变量名称而不是一个列名称”，从而让你灵活地用两个“命名空间”的资源（列名称的命名空间和 Python 对象的命名空间）计算代数式。需要注意的
是， @ 符号只能在 DataFrame.eval() 方法中使用，而不能在 pandas.eval() 函数中使用，因为 pandas.eval() 函数只能获取一个（Python）命名空间的内容。

### DataFrame.query()方法
```python
DataFrame 基于字符串代数式的运算实现了另一个方法，被称为 query()，例如：
In[23]: result1 = df[(df.A < 0.5) & (df.B < 0.5)]
result2 = pd.eval('df[(df.A < 0.5) & (df.B < 0.5)]')
np.allclose(result1, result2)
Out[23]: True
```
和前面介绍过的 DataFrame.eval() 一样，这是一个用 DataFrame 列创建的代数式，但是不能用 DataFrame.eval() 语法 

### 　性能决定使用时机

在考虑要不要用这两个函数时，需要思考两个方面： 计算时间和内存消耗，而内存消耗是更重要的影响因素。就像前面介绍的那样，每个涉及 NumPy 数组或 Pandas 的 DataFrame的复合代数式都会产生临时数组
In[26]: x = df[(df.A < 0.5) & (df.B < 0.5)]
它基本等价于：
```python
In[27]: tmp1 = df.A < 0.5
tmp2 = df.B < 0.5
tmp3 = tmp1 & tmp2
x = df[tmp3]
```
如果临时 DataFrame 的内存需求比你的系统内存还大（通常是几吉字节），那么最好还是使用 eval() 和 query() 代数式

##  Matplotlib常用技巧
### 　导入Matplotlib
就像之前用 np 作为 NumPy 的简写形式、 pd 作为 Pandas 的简写形式一样，我们也可以在导入 Matplotlib 时用一些它常用的简写形式：
```python
In[1]: import matplotlib as mpl
import matplotlib.pyplot as plt
```
plt 是最常用的接口，在本章后面的内容中会经常用到。

###　设置绘图样式

我们将使用 plt.style 来选择图形的绘图风格。现在选择经典（classic）风格，这样画出的图就都是经典的 Matplotlib 风格了：
```python
In[2]: plt.style.use('classic')
```
在后面的内容中，我们将根据需要调整绘图风格。 Matplotlib 在 1.5 版之后开始支持不同的风格列表（stylesheets）。如果你用的 Matplotlib 版本较旧，那么就只能使用默认的绘图风格。

### 　用不用show()？ 如何显示图形

如果数据可视化图不能被看见，那就一点儿用也没有了。但如何显示你的图形，就取决于具体的开发环境了。Matplotlib 的最佳实践与你使用的开发环境有关。简单来说，就是三种开发环境，分别是脚本、 IPython shell 和 IPython Notebook。
#### 1. 在脚本中画图
如果你在一个脚本文件中使用 Matplotlib，那么显示图形的时候必须使用 plt.show()。 plt.show() 会启动一个事件循环（event loop），并找到所有当前可用的图形对象，然后打开一
个或多个交互式窗口显示图形。
#### 2. 在IPython shell中画图
在 IPython shell 中交互式地使用 Matplotlib 画图非常方便（详情请参见第 1 章），在IPython 启动 Matplotlib 模式就可以使用它。为了启用这个模式，你需要在启动 ipython 后使用 %matplotlib 魔法命令
#### 3. 在IPython Notebook中画图
IPython Notebook 是一款基于浏览器的交互式数据分析工具，可以将描述性文字、代码、图形、 HTML 元素以及更多的媒体形式组合起来，集成到单个可执行的 Notebook 文档中
### 将图形保存为文件

Matplotlib 的一个优点是能够将图形保存为各种不同的数据格式。你可以用 savefig() 命令将图形保存为文件。例如，如果要将图形保存为 PNG 格式，你可以运行这行代码：
```python
In[5]: fig.savefig('my_figure.png')
这样工作文件夹里就有了一个 my_figure.png 文件：
In[6]: !ls -lh my_figure.png
-rw-r--r-- 1 jakevdp staff 16K Aug 11 10:59 my_figure.png
```
## 两种画图接口

不过 Matplotlib 有一个容易让人混淆的特性，就是它的两种画图接口：一个是便捷的MATLAB 风格接口，另一个是功能更强大的面向对象接口。下面来快速对比一下两种接口的主要差异。

### MATLAB风格接口

Matplotlib最初作为MATLAB用的Python替代品，许多语法都和MATLAB类似。MATLAB 风格的工具位于 pyplot（plt）接口中。 MATLAB 用户肯定对下面的代码特别熟悉
```python
In[9]: plt.figure()  创建图形
创建两个子图中的第一个，设置坐标轴
plt.subplot(2, 1, 1) # (行、列、子图编号)
plt.plot(x, np.sin(x))
创建两个子图中的第二个，设置坐标轴
plt.subplot(2, 1, 2)
plt.plot(x, np.cos(x));
```
#### 　面向对象接口

面向对象接口可以适应更复杂的场景，更好地控制你自己的图形。在面向对象接口中，画图函数不再受到当前“活动”图形或坐标轴的限制，而变成了显式的 Figure 和 Axes 的方法。通过下面的代码，可以用面向对象接口重新创建之前的图形（如图 4-4 所示）：
```python
In[10]: # 先创建图形网格
ax是一个包含两个Axes对象的数组
fig, ax = plt.subplots(2)
在每个对象上调用plot()方法
ax[0].plot(x, np.sin(x))
ax[1].plot(x, np.cos(x));
```
## 　简易线形图

在所有图形中，最简单的应该就是线性方程 y = f (x) 的可视化了。来看看如何创建这个简单的线形图。接下来的内容都是在 Notebook 中画图，因此需要导入以下命令：
```python
In[1]: %matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
要画 Matplotlib 图形时，都需要先创建一个图形 fig 和一个坐标轴 ax。创建图形与坐标轴的最简单做法如下所示（如图 4-5 所示）：
In[2]: fig = plt.figure()
ax = plt.axes()
```
### 　调整图形： 线条的颜色与风格

通常对图形的第一次调整是调整它线条的颜色与风格。 plt.plot() 函数可以通过相应的参
数设置颜色与风格。要修改颜色，就可以使用 color 参数，它支持各种颜色值的字符串。
颜色的不同表示方法如下所示（如图 4-9 所示）：
```python
In[6]:
plt.plot(x, np.sin(x - 0), color='blue') # 标准颜色名称
plt.plot(x, np.sin(x - 1), color='g') # 缩写颜色代码（rgbcmyk）
plt.plot(x, np.sin(x - 2), color='0.75') # 范围在0~1的灰度值
plt.plot(x, np.sin(x - 3), color='#FFDD44') # 十六进制（RRGGBB， 00~FF）
plt.plot(x, np.sin(x - 4), color=(1.0,0.2,0.3)) # RGB元组，范围在0~1
plt.plot(x, np.sin(x - 5), color='chartreuse'); # HTML颜色名称
```
### 　调整图形： 坐标轴上下限

虽然 Matplotlib 会自动为你的图形选择最合适的坐标轴上下限，但是有时自定义坐标轴上下限可能会更好。调整坐标轴上下限最基础的方法是 plt.xlim() 和 plt.ylim()（如图 4-12所示）：
```python
In[9]: plt.plot(x, np.sin(x))
plt.xlim(-1, 11)
plt.ylim(-1.5, 1.5);
还有一个方法是 plt.axis()（注意不要搞混 axes 和 axis）。通过传入 [xmin, xmax,ymin, ymax] 对应的值， plt.axis() 方法可以让你用一行代码设置 x 和 y 的限值
plt.axis() 能做的可不止如此，它还可以按照图形的内容自动收紧坐标轴，不留空白区域
```
### 　设置图形标签

在单个坐标轴上显示多条线时，创建图例显示每条线是很有效的方法。 Matplotlib 内置了一个简单快速的方法，可以用来创建图例，那就是（估计你也猜到了） plt.legend()。虽
然有不少用来设置图例的办法，但我觉得还是在 plt.plot 函数中用 label 参数为每条线设置一个标签最简单
你会发现， plt.legend() 函数会将每条线的标签与其风格、颜色自动匹配。关于通过 plt.legend() 设置图例的更多信息，请参考相应的程序文档。另外，我们将在 4.8 节介绍更多
高级的图例设置方法。
另一种常用的图形是简易散点图（scatter plot），与线形图类似。这种图形不再由线段连接，而是由独立的点、圆圈或其他形状构成。开始的时候同样需要在 Notebook 中导入函数：
```python
In[1]: %matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
```
### 　用plt.plot画散点图

上一节介绍了用 plt.plot/ax.plot 画线形图的方法，现在用这些函数来画散点图：
```python
In[2]: x = np.linspace(0, 10, 30)
y = np.sin(x)
plt.plot(x, y, 'o', color='black');
```
### 　用plt.scatter画散点图

另一个可以创建散点图的函数是 plt.scatter。它的功能非常强大，其用法与 plt.plot 函数类似（如图 4-24 所示）：
```python
In[6]: plt.scatter(x, y, marker='o')
```
plt.scatter 与 plt.plot 的主要差别在于，前者在创建散点图时具有更高的灵活性，可以单独控制每个散点与数据匹配，也可以让每个散点具有不同的属性（大小、表面颜色、边
框颜色等）。
下面来创建一个随机散点图，里面有各种颜色和大小的散点。为了能更好地显示重叠部分，用 alpha 参数来调整透明度（如图 4-25 所示）：
```python
In[7]: rng = np.random.RandomState(0)
x = rng.randn(100)
y = rng.randn(100)
colors = rng.rand(100)
sizes = 1000 * rng.rand(100)
plt.scatter(x, y, c=colors, s=sizes, alpha=0.3,
cmap='viridis')
plt.colorbar(); # 显示颜色条
```
## 　可视化异常处理

在数据可视化的结果中用图形将误差有效地显示出来，就可以提供更充分的信息。

### 　基本误差线

基本误差线（errorbar）可以通过一个 Matplotlib 函数来创建
```python
In[1]: %matplotlib inline
       import matplotlib.pyplot as plt
       plt.style.use('seaborn-whitegrid')
       import numpy as np
In[2]: x = np.linspace(0, 10, 50)
       dy = 0.8
       y = np.sin(x) + dy * np.random.randn(50)
       plt.errorbar(x, y, yerr=dy, fmt='.k');
```
### 连续误差

有时候可能需要显示连续变量的误差。虽然 Matplotlib 没有内置的简便方法可以解决这个问题，但是通过 plt.plot 与 plt.fill_between 来解决也不是很难。
我们将用 Scikit-Learn 程序库 API 里面一个简单的高斯过程回归方法（Gaussian processregression， GPR）来演示。这是用一种非常灵活的非参数方程（nonparametric function）对带有不确定性的连续测量值进行拟合的方法。
```python
In[4]: from sklearn.gaussian_process import GaussianProcess
定义模型和要画的数据
model = lambda x: x * np.sin(x)
xdata = np.array([1, 3, 5, 6, 8])
ydata = model(xdata)
计算高斯过程拟合结果
gp = GaussianProcess(corr='cubic', theta0=1e-2, thetaL=1e-4, thetaU=1E-1,
random_start=100)
gp.fit(xdata[:, np.newaxis], ydata)
xfit = np.linspace(0, 10, 1000)
yfit, MSE = gp.predict(xfit[:, np.newaxis], eval_MSE=True)
dyfit = 2 * np.sqrt(MSE) # 2*sigma~95%置信区间
现在，我们获得了 xfit、 yfit 和 dyfit，表示数据的连续拟合结果。接着，如上所示将这些数据传入 plt.errorbar 函数。但是我们并不是真的要为 1000 个数据点画上 1000 条误差线；相反，可以通过在 plt.fill_between 函数中设置颜色来表示连续误差线
In[5]: # 将结果可视化
plt.plot(xdata, ydata, 'or')
plt.plot(xfit, yfit, '-', color='gray')
plt.fill_between(xfit, yfit - dyfit, yfit + dyfit,
color='gray', alpha=0.2)
plt.xlim(0, 10);
```
请注意，我们将 fill_between 函数设置为：首先传入 x 轴坐标值，然后传入 y 轴下边界以及 y 轴上边界，这样整个区域就被误差线填充了。
从结果图形中可以非常直观地看出高斯过程回归方法拟合的效果：在接近样本点的区域，模型受到很强的约束，拟合误差非常小，非常接近真实值；而在远离样本点的区域，模型
不受约束，误差不断增大。若想获取更多关于 plt.fill_between() 函数（以及它与 plt.fill() 的紧密关系）选项的信息，请参考函数文档或者 Matplotlib 文档。
最后提一点，如果你觉得这样实现连续误差线的做法太原始，可以参考 4.16 节，我们会在那里介绍 Seaborn 程序包，它提供了一个更加简便的 API 来实现连续误差线。
## 　密度图与等高线图

有时在二维图上用等高线图或者彩色图来表示三维数据是个不错的方法。 Matplotlib 提供了三个函数来解决这个问题：用 plt.contour 画等高线图、用 plt.contourf 画带有填充色
的等高线图（filled contour plot）的色彩、用 plt.imshow 显示图形。
三维函数的可视化
首先用函数 z = f (x, y) 演示一个等高线图，按照下面的方式生成函数 f（在 2.5 节已经介绍过，当时用它来演示数组的广播功能）样本数据：
```python
In[2]: def f(x, y):
return np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)
```
等高线图可以用 plt.contour 函数来创建。它需要三个参数： x 轴、 y 轴、 z 轴三个坐标轴的网格数据。 x 轴与 y 轴表示图形中的位置，而 z 轴将通过等高线的等级来表示。用
np.meshgrid 函数来准备这些数据可能是最简单的方法，它可以从一维数组构建二维网格数据：
```python
In[3]: x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 40)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)
```
现在来看看标准的线形等高线图（如图 4-30 所示）：
In[4]: plt.contour(X, Y, Z, colors='black')
需要注意的是，当图形中只使用一种颜色时，默认使用虚线表示负数，使用实线表示正数。另外，你可以用 cmap 参数设置一个线条配色方案来自定义颜色。还可以让更多的线
条显示不同的颜色——可以将数据范围等分为 20 份，然后用不同的颜色表示：
In[5]: plt.contour(X, Y, Z, 20, cmap='RdGy');
现在使用 RdGy（红 - 灰， Red-Gray 的缩写）配色方案，这对于数据集中度的显示效果比较好。 Matplotlib 有非常丰富的配色方案，你可以在 IPython 里用 Tab 键浏览 plt.cm 模块对应的信息：
plt.cm.<TAB>
虽然这幅图看起来漂亮多了，但是线条之间的间隙还是有点大 。我们可以通过 plt.contourf()
函数来填充等高线图（需要注意结尾有字母 f），它的语法和 plt.contour() 是一样的。
另外还可以通过 plt.colorbar() 命令自动创建一个表示图形各种颜色对应标签信息的颜色
条：
```python
In[6]: plt.contourf(X, Y, Z, 20, cmap='RdGy')
       plt.colorbar();
 ```
• plt.imshow() 不支持用 x 轴和 y 轴数据设置网格，而是必须通过 extent 参数设置图形
的坐标范围 [xmin, xmax, ymin, ymax]。
• plt.imshow() 默认使用标准的图形数组定义，就是原点位于左上角（浏览器都是如此），
而不是绝大多数等高线图中使用的左下角。这一点在显示网格数据图形的时候必须调整。
• plt.imshow() 会 自 动 调 整 坐 标 轴 的 精 度 以 适 应 数 据 显 示。 你 可 以 通 过 plt.
axis(aspect='image') 来设置 x 轴与 y 轴的单位。

