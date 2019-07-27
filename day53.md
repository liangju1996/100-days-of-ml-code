## Matplotlib自定义：配置文件与样式表
Matplotlib 的默认图形设置经常被用户诟病。虽然 Matplotlib 2.0 版本已经有大幅改善，但
是掌握自定义配置的方法可以让我们打造自己的艺术风格。
首先简单浏览一下 Matplotlib 的运行时配置（runtime configuration，rc）功能的介绍，然后
再看看新式的样式表（stylesheets）特性，里面包含了许多漂亮的默认配置功能。
### 手动配置图形
通过本章的介绍，我们已经知道如何修改单个图形配置，使得最终图形比原来的图形更好
看。可以为每个单独的图形进行个性化设置。举个例子，看看由下面这个土到掉渣的默认
配置生成的频次直方图：
```python
In[1]: import matplotlib.pyplot as plt
 plt.style.use('classic')
 import numpy as np
 %matplotlib inline
In[2]: x = np.random.randn(1000)
 plt.hist(x);
```
通过手动调整，可以让它成为美图：
```python
In[3]: # 用灰色背景
 ax = plt.axes(axisbg='#E6E6E6')
 ax.set_axisbelow(True)
 # 画上白色的网格线
 plt.grid(color='w', linestyle='solid')
 # 隐藏坐标轴的线条
 for spine in ax.spines.values():
 spine.set_visible(False) 
Matplotlib数据可视化 ｜ 249
 # 隐藏上边与右边的刻度
 ax.xaxis.tick_bottom()
 ax.yaxis.tick_left()
 # 弱化刻度与标签
 ax.tick_params(colors='gray', direction='out')
 for tick in ax.get_xticklabels():
 tick.set_color('gray')
 for tick in ax.get_yticklabels():
 tick.set_color('gray')
 # 设置频次直方图轮廓色与填充色
 ax.hist(x, edgecolor='#E6E6E6', color='#EE6666');
```
这样看起来就漂亮多了。你可能会觉得它的风格与 R 语言的 ggplot 可视化程序包有点儿
像。但这样设置可太费劲儿了！我们肯定不希望每做一个图都需要这样手动配置一番。好
在已经有一种方法，可以让我们只配置一次默认图形，就能将其应用到所有图形上。
### 修改默认配置：rcParams
Matplotlib 每次加载时，都会定义一个运行时配置（rc），其中包含了所有你创建的图形元
素的默认风格。你可以用 plt.rc 简便方法随时修改这个配置。来看看如何调整 rc 参数，
用默认图形实现之前手动调整的效果。
先复制一下目前的 rcParams 字典，这样可以在修改之后再还原回来：
```python
In[4]: IPython_default = plt.rcParams.copy()
现在就可以用 plt.rc 函数来修改配置参数了：
In[5]: from matplotlib import cycler
 colors = cycler('color',
 ['#EE6666', '#3388BB', '#9988DD',
 '#EECC55', '#88BB44', '#FFBBBB'])
 plt.rc('axes', facecolor='#E6E6E6', edgecolor='none',
 axisbelow=True, grid=True, prop_cycle=colors)

 plt.rc('grid', color='w', linestyle='solid')
 plt.rc('xtick', direction='out', color='gray')
 plt.rc('ytick', direction='out', color='gray')
 plt.rc('patch', edgecolor='#E6E6E6')
 plt.rc('lines', linewidth=2)
 ```
设置完成之后，来创建一个图形看看效果：

'In[6]: plt.hist(x);'


再画一些线图看看 rc 参数的效果：
```python
In[7]: for i in range(4):
 plt.plot(np.random.rand(10))
```
新的艺术风格比之前的默认风格更漂亮了。如果你不认同我的审美风格，当然可以自
己调整 rc 参数，创造自己的风格！这些设置会保存在 .matplotlibrc 文件中，你可以在
Matplotlib 文档（http://matplotlib.org/users/customizing.html）中找到更多信息。这时有人说
了，他们更喜欢自定义 Matplotlib 的样式表。

### 样式表
2014 年 8 月发布的 Matplotlib 1.4 版本中增加了一个非常好用的 style 模块，里面包含了大
量的新式默认样式表，还支持创建和打包你自己的风格。虽然这些样式表实现的格式功能
与前面介绍的 .matplotlibrc 文件类似，但是它的文件扩展名是 .mplstyle。
即使你不打算创建自己的绘图风格，样式表包含的默认内容也非常有用。通过 plt.style.
available 命令可以看到所有可用的风格，下面将简单介绍前五种风格：
```python
In[8]: plt.style.available[:5]
Out[8]: ['fivethirtyeight',
 'seaborn-pastel',
 'seaborn-whitegrid',
 'ggplot',
 'grayscale']
 ```
使用某种样式表的基本方法如下所示：

'plt.style.use('stylename')'

但需要注意的是，这样会改变后面所有的风格！如果需要，你可以使用风格上下文管理器
（context manager）临时更换至另一种风格：
```python
with plt.style.context('stylename'):
 make_a_plot()
 ```
来创建一个可以画两种基本图形的函数：
```python
In[9]: def hist_and_lines():
 np.random.seed(0)
 fig, ax = plt.subplots(1, 2, figsize=(11, 4))
 ax[0].hist(np.random.randn(1000))
 for i in range(3):
 ax[1].plot(np.random.rand(10))
 ax[1].legend(['a', 'b', 'c'], loc='lower left')
 ```
下面就用这个函数来演示不同风格的显示效果。
#### 1. 默认风格
默认风格就是本书前面内容中一直使用的风格，我们就从这里开始。首先，将之前设置的
运行时配置还原为默认配置：
```python
In[10]: # 重置rcParams
 plt.rcParams.update(IPython_default);
 ```
现在来看看默认风格的效果：  

'In[11]: hist_and_lines()'

#### 2. FiveThirtyEight风格
FiveThirtyEight 风格模仿的是著名网站 FiveThirtyEight（http://fivethirtyeight.com）的绘图
风格。这种风格使用深色的粗线条和透明的坐标轴：
```python
In[12]: with plt.style.context('fivethirtyeight'):
 hist_and_lines()
```
#### 3. ggplot风格
R 语言的 ggplot 是非常流行的可视化工具，Matplotlib 的 ggplot 风格就是模仿这个程序包
的默认风格：
```python
In[13]: with plt.style.context('ggplot'):
 hist_and_lines()
```
#### 4. bmh风格
有一本短小精悍的在线图书叫 Probabilistic Programming and Bayesian Methods for Hackers
（http://bit.ly/2fDJsKC）。整本书的图形都是用 Matplotlib 创建的，通过一组 rc 参数创建了
一种引人注目的绘图风格。这个风格被 bmh 样式表继承了：
```python
In[14]: with plt.style.context('bmh'):
 hist_and_lines()
```
#### 5. 黑色背景风格
在演示文档中展示图形时，用黑色背景而非白色背景往往会取得更好的效果。dark_
background 风格就是为此设计的：
```python
In[15]: with plt.style.context('dark_background'):
 hist_and_lines()
```
#### 6. 灰度风格
有时你可能会做一些需要打印的图形，不能使用彩色。这时使用 grayscale 风格的效果最好：
```python
In[16]: with plt.style.context('grayscale'):
 hist_and_lines()
```
#### 7. Seaborn风格
Matplotlib 还有一些灵感来自 Seaborn 程序库（将在 4.16 节详细介绍）的风格，这些风格
在 Notebook 导入 Seaborn 程序库后会自动加载。我觉得这些风格非常漂亮，也是我自己在
探索数据时一直使用的默认风格：
```python
In[17]: import seaborn
 hist_and_lines()
```
通过运用各式各样的内置绘图风格，Matplotlib 在交互式可视化与创建印刷品图形两方面
都表现得越来越好。在创建这本书的图形时，我通常会用一种或几种内置的绘图风格。
