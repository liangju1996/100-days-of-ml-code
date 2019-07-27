## 频次直方图、数据区间划分和分布密度
只要导入了画图的函数，只用一行代 码就可以创建一个简易的频次直方图（如图 4-35 所示）：
```python
In[1]: %matplotlib inline       
import numpy as np        
import matplotlib.pyplot as plt        
plt.style.use('seaborn-white') 
 
data = np.random.randn(1000) 
 
In[2]: plt.hist(data);
```
hist() 有许多用来调整计算过程和显示效果的选项
```python
In[3]: plt.hist(data, bins=30, normed=True, alpha=0.5,
 histtype='stepfilled', color='steelblue',
 edgecolor='none');
```
关于 plt.hist 自定义选项的更多内容都在它的程序文档中。我发现在用频次直方图对不同
分布特征的样本进行对比时，将 histtype='stepfilled' 与透明性设置参数 alpha 搭配使用
的效果非常好
```python
In[4]: x1 = np.random.normal(0, 0.8, 1000)
 x2 = np.random.normal(-2, 1, 1000)
 x3 = np.random.normal(3, 2, 1000)
 kwargs = dict(histtype='stepfilled', alpha=0.3, normed=True, bins=40)
 plt.hist(x1, **kwargs)
 plt.hist(x2, **kwargs)
 plt.hist(x3, **kwargs);
```

如果你只需要简单地计算频次直方图（就是计算每段区间的样本数），而并不想画图显示
它们，那么可以直接用 np.histogram()：
```python
In[5]: counts, bin_edges = np.histogram(data, bins=5)
 print(counts)
[ 12 190 468 301 29]
```
### 1.二维频次直方图与数据区间划分
也可以将二维数组按照二维区间进行切分，来创建二维频次直方图。
```python
In[6]: mean = [0, 0]
 cov = [[1, 1], [1, 2]]
 x, y = np.random.multivariate_normal(

In[12]: plt.hist2d(x, y, bins=30, cmap='Blues')
 cb = plt.colorbar()
 cb.set_label('counts in bin')
```
与 plt.hist 函数一样，plt.hist2d 也有许多调整图形与区间划分的配置选项，详细内容都
在程序文档中。另外，就像 plt.hist 有一个只计算结果不画图的 np.histogram 函数一样，
plt.hist2d 类似的函数是 np.histogram2d，其用法如下所示：
```python
In[8]: counts, xedges, yedges = np.histogram2d(x, y, bins=30)
```

### 2. plt.hexbin：六边形区间划分
二维频次直方图是由与坐标轴正交的方块分割而成的，还有一种常用的方式是用正六边
形分割。Matplotlib 提供了 plt.hexbin 满足此类需求，将二维数据集分割成蜂窝状
```python
In[9]: plt.hexbin(x, y, gridsize=30, cmap='Blues')
 cb = plt.colorbar(label='count in bin')

plt.hexbin 同样也有一大堆有趣的配置选项，包括为每个数据点设置不同的权重，以及用
任意 NumPy 累计函数改变每个六边形区间划分的结果（权重均值、标准差等指标）。
### 3. 核密度估计
还有一种评估多维数据分布密度的常用方法是核密度估计（kernel density estimation，
KDE）。我们将在 5.13 节详细介绍这种方法，现在先来简单地演示如何用 KDE 方法“抹
掉”空间中离散的数据点，从而拟合出一个平滑的函数。在 scipy.stats 程序包里面有一
个简单快速的 KDE 实现方法，下面就是用这个方法演示的简单示例
```python
In[10]: from scipy.stats import gaussian_kde
 # 拟合数组维度[Ndim, Nsamples]
 data = np.vstack([x, y])
 kde = gaussian_kde(data)
 # 用一对规则的网格数据进行拟合
 xgrid = np.linspace(-3.5, 3.5, 40)
 ygrid = np.linspace(-6, 6, 40)
 Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
 Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
 # 画出结果图
 plt.imshow(Z.reshape(Xgrid.shape),
 origin='lower', aspect='auto',
 extent=[-3.5, 3.5, -6, 6],
 cmap='Blues')
 cb = plt.colorbar()
 cb.set_label("density")
```
KDE 方法通过不同的平滑带宽长度（smoothing length）在拟合函数的准确性与平滑性之
间作出权衡（无处不在的偏差与方差的取舍问题的一个例子）。想找到恰当的平滑带宽长
度是件很困难的事，gaussian_kde 通过一种经验方法试图找到输入数据平滑长度的近似
最优解。

在 SciPy 的生态系统中还有其他的 KDE 方法实现，每种版本都有各自的优缺点，例
如 sklearn.neighbors.KernelDensity 和 statsmodels.nonparametric.kernel_density.
KDEMultivariate。用 Matplotlib 做 KDE 的可视化图的过程比较繁琐，Seaborn 程序库（详
情请参见 4.16 节）提供了一个更加简洁的 API 来创建基于 KDE 的可视化图。

## 配置图例
想在可视化图形中使用图例，可以为不同的图形元素分配标签。前面介绍过如何创建简单
的图例，现在将介绍如何在 Matplotlib 中自定义图例的位置与艺术风格。
可以用 plt.legend() 命令来创建最简单的图例，它会自动创建一个包含每个图形元素的图
例：
```python
In[1]: import matplotlib.pyplot as plt
 plt.style.use('classic')
In[2]: %matplotlib inline
 import numpy as np
In[3]: x = np.linspace(0, 10, 1000)
 fig, ax = plt.subplots()
 ax.plot(x, np.sin(x), '-b', label='Sine')
 ax.plot(x, np.cos(x), '--r', label='Cosine')
 ax.axis('equal')
 leg = ax.legend();
 ```

但是，我们经常需要对图例进行各种个性化的配置。例如，我们想设置图例的位置，并取
消外边框：
```python
In[4]: ax.legend(loc='upper left', frameon=False)
 fig
 ```
还可以用 ncol 参数设置图例的标签列数（如图 4-43 所示）：
In[5]: ax.legend(frameon=False, loc='lower center', ncol=2)
 fig
```
还可以为图例定义圆角边框（fancybox）、增加阴影、改变外边框透明度（framealpha 值），
或者改变文字间距：
```python
In[6]: ax.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)
 fig
```
### 选择图例显示的元素
我们已经看到，图例会默认显示所有元素的标签。如果你不想显示全部，可以通过一些图
形命令来指定显示图例中的哪些元素和标签。plt.plot() 命令可以一次创建多条线，返回
线条实例列表。一种方法是将需要显示的线条传入 plt.legend()，另一种方法是只为需要
在图例中显示的线条设置标签：
```python
In[7]: y = np.sin(x[:, np.newaxis] + np.pi * np.arange(0, 2, 0.5))
 lines = plt.plot(x, y)
 # lines变量是一组plt.Line2D实例
 plt.legend(lines[:2], ['first', 'second']);
```
在实践中，我发现第一种方法更清晰。当然也可以只为需要在图例中显示的元素设置标签：
```python
In[8]: plt.plot(x, y[:, 0], label='first')
 plt.plot(x, y[:, 1], label='second')
 plt.plot(x, y[:, 2:])
 plt.legend(framealpha=1, frameon=True);
```
### 在图例中显示不同尺寸的点
有时，默认的图例仍然不能满足我们的可视化需求。例如，你可能需要用不同尺寸的点来
表示数据的特征，并且希望创建这样的图例来反映这些特征。下面的示例将用点的尺寸来
表明美国加州不同城市的人口数量。如果我们想要一个通过不同尺寸的点显示不同人口数
量级的图例，可以通过隐藏一些数据标签来实现这个效果：
```python
In[9]: import pandas as pd
 cities = pd.read_csv('data/california_cities.csv')
 # 提取感兴趣的数据
 lat, lon = cities['latd'], cities['longd']
 population, area = cities['population_total'], cities['area_total_km2']
 # 用不同尺寸和颜色的散点图表示数据，但是不带标签
 plt.scatter(lon, lat, label=None,
 c=np.log10(population), cmap='viridis',
 s=area, linewidth=0, alpha=0.5)
 plt.axis(aspect='equal')
 plt.xlabel('longitude')
 plt.ylabel('latitude')
 plt.colorbar(label='log$_{10}$(population)')
 plt.clim(3, 7)
 # 下面创建一个图例：
 # 画一些带标签和尺寸的空列表
 for area in [100, 300, 500]:
 plt.scatter([], [], c='k', alpha=0.3, s=area,
 label=str(area) + ' km$^2$')
 plt.legend(scatterpoints=1, frameon=False,
 labelspacing=1, title='City Area')
 plt.title('California Cities: Area and Population');
```
由于图例通常是图形中对象的参照，因此如果我们想显示某种形状，就需要将它画出来。
但是在这个示例中，我们想要的对象（灰色圆圈）并不在图形中，因此把它们用空列表假
装画出来。还需要注意的是，图例只会显示带标签的元素。
为了画出这些空列表中的图形元素，需要为它们设置标签，以便图例可以显示它们，这样
就可以从图例中获得想要的信息了。这个策略对于创建复杂的可视化图形很有效。
最后需要注意的是，在处理这类地理数据的时候，如果能把州的地理边界或其他地图元素
也显示出来，那么图形就会更加逼真。Matplotlib 的 Basemap（底图）插件工具箱恰好是做
这种事情的最佳选择，我们将在 4.15 节介绍它。
### 同时显示多个图例
有时，我们可能需要在同一张图上显示多个图例。不过，用 Matplotlib 解决这个问题并不
容易，因为通过标准的 legend 接口只能为一张图创建一个图例。如果你想用 plt.legend()
或 ax.legend() 方法创建第二个图例，那么第一个图例就会被覆盖。但是，我们可以通
过从头开始创建一个新的图例艺术家对象（legend artist），然后用底层的（lower-level）
ax.add_artist() 方法在图上添加第二个图例：
```python
In[10]: fig, ax = plt.subplots()
 lines = []
 styles = ['-', '--', '-.', ':']
 x = np.linspace(0, 10, 1000)
 for i in range(4):
 lines += ax.plot(x, np.sin(x - i * np.pi / 2),
 styles[i], color='black')
 ax.axis('equal')
 # 设置第一个图例要显示的线条和标签
 ax.legend(lines[:2], ['line A', 'line B'],
 loc='upper right', frameon=False)
 # 创建第二个图例，通过add_artist方法添加到图上
 from matplotlib.legend import Legend
 leg = Legend(ax, lines[2:], ['line C', 'line D'],
 loc='lower right', frameon=False)
 ax.add_artist(leg);
```
这里只是小试了一下构成 Matplotlib 图形的底层图例艺术家对象。如果你查看过 ax.legend()
的源代码（前面介绍过，在 IPython Notebook 里面用 ax.legend?? 来显示源代码），就会发
现这个函数通过几条简单的逻辑就创建了一个 Legend 图例艺术家对象，然后被保存到了
legend_ 属性里。当图形被画出来之后，就可以将该图例增加到图形上。
## 配置颜色条
图例通过离散的标签表示离散的图形元素。然而，对于图形中由彩色的点、线、面构成的
连续标签，用颜色条来表示的效果比较好。在 Matplotlib 里面，颜色条是一个独立的坐标
轴，可以指明图形中颜色的含义。首先还是导入需要使用的画图工具：
```python
In[1]: import matplotlib.pyplot as plt
 plt.style.use('classic')
In[2]: %matplotlib inline
 import numpy as np
 ```
和在前面看到的一样，通过 plt.colorbar 函数就可以创建最简单的颜色条
```python
In[3]: x = np.linspace(0, 10, 1000)
 I = np.sin(x) * np.cos(x[:, np.newaxis])
 plt.imshow(I)
 plt.colorbar();
```

### 配置颜色条
可以通过 cmap 参数为图形设置颜色条的配色方案
```python
In[4]: plt.imshow(I, cmap='gray');
```
所有可用的配色方案都在 plt.cm 命名空间里面，在 IPython 里通过 Tab 键就可以查看所有
的配置方案：
'plt.cm.<TAB>'
有了这么多能够作为备选的配色方案只是第一步，更重要的是如何确定用哪种方案！最终
的选择结果可能和你一开始想用的有很大不同。
#### 1. 选择配色方案
关于可视化图形颜色选择的全部知识超出了本书的介绍范围，但如果你想了解与此相关
的入门知识，可以参考文章“Ten Simple Rules for Better Figures”（http://bit.ly/2fDJn9J）。
Matplotlib 的在线文档中也有关于配色方案选择的有趣论述（http://matplotlib.org/1.4.1/
users/colormaps.html）。
一般情况下，你只需要重点关注三种不同的配色方案。
'顺序配色方案'
由一组连续的颜色构成的配色方案（例如 binary 或 viridis）。
'互逆配色方案'
通常由两种互补的颜色构成，表示正反两种含义（例如 RdBu 或 PuOr）。
'定性配色方案'
随机顺序的一组颜色（例如 rainbow 或 jet）。
'jet '是一种定性配色方案，曾是 Matplotlib 2.0 之前所有版本的默认配色方案。把它作为默
认配色方案实在不是个好主意，因为定性配色方案在对定性数据进行可视化时的选择空间
非常有限。随着图形亮度的提高，经常会出现颜色无法区分的问题。
可以通过把 jet 转换为黑白的灰度图看看具体的颜色：
```python
In[5]:
from matplotlib.colors import LinearSegmentedColormap
def grayscale_cmap(cmap): 

 """为配色方案显示灰度图"""
 cmap = plt.cm.get_cmap(cmap)
 colors = cmap(np.arange(cmap.N))
 # 将RGBA色转换为不同亮度的灰度值
 # 参考链接http://alienryderflex.com/hsp.html
 RGB_weight = [0.299, 0.587, 0.114]
 luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
 colors[:, :3] = luminance[:, np.newaxis]
 return LinearSegmentedColormap.from_list(cmap.name + "_gray", colors, cmap.N)
def view_colormap(cmap):
 """用等价的灰度图表示配色方案"""
 cmap = plt.cm.get_cmap(cmap)
 colors = cmap(np.arange(cmap.N))
 cmap = grayscale_cmap(cmap)
 grayscale = cmap(np.arange(cmap.N))
 fig, ax = plt.subplots(2, figsize=(6, 2),
 subplot_kw=dict(xticks=[], yticks=[]))
 ax[0].imshow([colors], extent=[0, 10, 0, 1])
 ax[1].imshow([grayscale], extent=[0, 10, 0, 1])
In[6]: view_colormap('jet')
```
注意观察灰度图里比较亮的那部分条纹。这些亮度变化不均匀的条纹在彩色图中对应某一
段彩色区间，由于色彩太接近容易突显出数据集中不重要的部分，导致眼睛无法识别重
点。更好的配色方案是 viridis（已经成为 Matplotlib 2.0 的默认配色方案）。它采用了精心
设计的亮度渐变方式，这样不仅便于视觉观察，而且转换成灰度图后也更清晰：  
'In[7]: view_colormap('viridis')'

如果你喜欢彩虹效果，可以用 cubehelix 配色方案来可视化连续的数值：  
'In[8]: view_colormap('cubehelix')'

至于其他的场景，例如要用两种颜色表示正反两种含义时，可以使用 RdBu 双色配色方案
（红色 - 蓝色，Red-Blue 简称）。用红色、蓝色表示的正反两种信息
在灰度图上看不出差别！  

'In[9]: view_colormap('RdBu')'

我们将在后面的章节中继续使用这些配色方案。
Matplotlib 里面有许多配色方案，在 IPython 里面用 Tab 键浏览 plt.cm 模块就可以看到所
有内容。关于 Python 语言中配色的更多基本原则，可以参考 Seaborn 程序库的工具和文档
（详情请参 4.16 节）。
#### 2. 颜色条刻度的限制与扩展功能的设置
Matplotlib 提供了丰富的颜色条配置功能。由于可以将颜色条本身仅看作是一个 plt.Axes
实例，因此前面所学的所有关于坐标轴和刻度值的格式配置技巧都可以派上用场。颜色条
有一些有趣的特性。例如，我们可以缩短颜色取值的上下限，对于超出上下限的数据，通
过 extend 参数用三角箭头表示比上限大的数或者比下限小的数。这种方法很简单，比如你
想展示一张噪点图：
```python
In[10]: # 为图形像素设置1%噪点
 speckles = (np.random.random(I.shape) < 0.01)
 I[speckles] = np.random.normal(0, 3, np.count_nonzero(speckles))
 plt.figure(figsize=(10, 3.5))
 plt.subplot(1, 2, 1)
 plt.imshow(I, cmap='RdBu')
 plt.colorbar()
 plt.subplot(1, 2, 2) 

 plt.imshow(I, cmap='RdBu')
 plt.colorbar(extend='both')
 plt.clim(-1, 1);
```
左边那幅图是用默认的颜色条刻度限制实现的效果，噪点的范围完全覆盖了我们感兴趣的
数据。而右边的图形设置了颜色条的刻度上下限，并在上下限之外增加了扩展功能，这样
的数据可视化图形显然更有效果。
#### 3. 离散型颜色条
虽然颜色条默认都是连续的，但有时你可能也需要表示离散数据。最简单的做法就是使用
plt.cm.get_cmap() 函数，将适当的配色方案的名称以及需要的区间数量传进去即可
```python
In[11]: plt.imshow(I, cmap=plt.cm.get_cmap('Blues', 6))
 plt.colorbar()
 plt.clim(-1, 1);
```
这种离散型颜色条和其他颜色条的用法相同。
### 案例：手写数字
让我们来看一些有趣的手写数字可视化图，这可能是一个比较实用的案例。数据在 Scikit-
Learn 里面，包含近 2000 份 8×8 的手写数字缩略图。
先下载数据，然后用 plt.imshow() 对一些图形进行可视化:
```python
In[12]: # 加载数字0~5的图形，对其进行可视化
 from sklearn.datasets import load_digits
 digits = load_digits(n_class=6)
 fig, ax = plt.subplots(8, 8, figsize=(6, 6))
 for i, axi in enumerate(ax.flat):
 axi.imshow(digits.images[i], cmap='binary')
 axi.set(xticks=[], yticks=[])
```
由于每个数字都由 64 像素的色相（hue）构成，因此可以将每个数字看成是一个位于 64
维空间的点，即每个维度表示一个像素的亮度。但是想通过可视化来描述如此高维度的空
间是非常困难的。一种解决方案是通过降维技术，在尽量保留数据内部重要关联性的同时
降低数据的维度，例如流形学习（manifold learning）。降维是无监督学习的重要内容，5.1
节将详细介绍这部分知识。
暂且不提具体的降维细节，先来看看如何用流形学习将这些数据投影到二维空间进行可视
化（详情请参见 5.10 节）：
```python
In[13]: # 用IsoMap方法将数字投影到二维空间
 from sklearn.manifold import Isomap
 iso = Isomap(n_components=2)
 projection = iso.fit_transform(digits.data)
 ```
我们将用离散型颜色条来显示结果，调整 ticks 与 clim 参数来改善颜色条：
```python
In[14]: # 画图
 plt.scatter(projection[:, 0], projection[:, 1], lw=0.1,
 c=digits.target, cmap=plt.cm.get_cmap('cubehelix', 6))
 plt.colorbar(ticks=range(6), label='digit value')
 plt.clim(-0.5, 5.5)
```
这个投影结果还向我们展示了一些数据集的有趣特性。例如，数字 5 与数字 3 在投影中有
大面积重叠，说明一些手写的 5 与 3 难以区分，因此自动分类算法也更容易搞混它们。其
他的数字，像数字 0 与数字 1，隔得特别远，说明两者不太可能出现混淆。这个观察结果
也符合我们的直观感受，因为 5 和 3 看起来确实要比 0 和 1 更像。

4.10　多子图
有时候需要从多个角度对数据进行对比。Matplotlib 为此提出了子图（subplot）的概念：在
较大的图形中同时放置一组较小的坐标轴。这些子图可能是画中画（inset）、网格图（grid
of plots），或者是其他更复杂的布局形式。在这一节中，我们将介绍四种用 Matplotlib 创建
子图的方法。首先，在 Notebook 中导入画图需要的程序库：
In[1]: %matplotlib inline
 import matplotlib.pyplot as plt
 plt.style.use('seaborn-white')
 import numpy as np
4.10.1 plt.axes：手动创建子图
创建坐标轴最基本的方法就是使用 plt.axes 函数。前面已经介绍过，这个函数的默认配置
是创建一个标准的坐标轴，填满整张图。它还有一个可选参数，由图形坐标系统的四个值
构成。这四个值分别表示图形坐标系统的 [bottom, left, width, height]（底坐标、左坐
标、宽度、高度），数值的取值范围是左下角（原点）为 0，右上角为 1。
如果想要在右上角创建一个画中画，那么可以首先将 x 与 y 设置为 0.65（就是坐标轴原点
位于图形高度 65% 和宽度 65% 的位置），然后将 x 与 y 扩展到 0.2（也就是将坐标轴的宽
度与高度设置为图形的 20%）。图 4-59 显示了代码的结果：
In[2]: ax1 = plt.axes() # 默认坐标轴
 ax2 = plt.axes([0.65, 0.65, 0.2, 0.2])
图灵社区会员 felix123(490049061@qq.com) 专享 尊重版权
Matplotlib数据可视化 ｜ 231
图 4-59：图中图的坐标轴
面向对象画图接口中类似的命令有 fig.add_axes()。用这个命令创建两个竖直排列的坐标
轴（如图 4-60 所示）：
In[3]: fig = plt.figure()
 ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4],
 xticklabels=[], ylim=(-1.2, 1.2))
 ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4],
 ylim=(-1.2, 1.2))
 x = np.linspace(0, 10)
 ax1.plot(np.sin(x))
 ax2.plot(np.cos(x));
图 4-60：竖直排列的坐标轴
现在就可以看到两个紧挨着的坐标轴（上面的坐标轴没有刻度）：上子图（起点 y 坐标为
0.5 位置）与下子图的 x 轴刻度是对应的（起点 y 坐标为 0.1，高度为 0.4）。
4.10.2 plt.subplot：简易网格子图
若干彼此对齐的行列子图是常见的可视化任务，Matplotlib 拥有一些可以轻松创建它们的
简便方法。最底层的方法是用 plt.subplot() 在一个网格中创建一个子图。这个命令有三
232 ｜ 第 4 章
个整型参数——将要创建的网格子图行数、列数和索引值，索引值从 1 开始，从左上角到
右下角依次增大（如图 4-61 所示）：
In[4]: for i in range(1, 7):
 plt.subplot(2, 3, i)
 plt.text(0.5, 0.5, str((2, 3, i)),
 fontsize=18, ha='center')
图 4-61：plt.subplot()
plt.subplots_adjust 命令可以调整子图之间的间隔。用面向对象接口的命令 fig.add_
subplot() 可以取得同样的效果（结果如图 4-62 所示）：
In[5]: fig = plt.figure()
 fig.subplots_adjust(hspace=0.4, wspace=0.4)
 for i in range(1, 7):
 ax = fig.add_subplot(2, 3, i)
 ax.text(0.5, 0.5, str((2, 3, i)),
 fontsize=18, ha='center')
图 4-62：带边距调整功能的 plt.subplot()
我们通过 plt.subplots_adjust 的 hspace 与 wspace 参数设置与图形高度与宽度一致的子图
间距，数值以子图的尺寸为单位（在本例中，间距是子图宽度与高度的 40%）。
Matplotlib数据可视化 ｜ 233
4.10.3 plt.subplots：用一行代码创建网格
当你打算创建一个大型网格子图时，就没办法使用前面那种亦步亦趋的方法了，尤其是当
你想隐藏内部子图的 x 轴与 y 轴标题时。出于这一需求，plt.subplots() 实现了你想要的
功能（需要注意此处 subplots 结尾多了个 s）。这个函数不是用来创建单个子图的，而是
用一行代码创建多个子图，并返回一个包含子图的 NumPy 数组。关键参数是行数与列数，
以及可选参数 sharex 与 sharey，通过它们可以设置不同子图之间的关联关系。
我们将创建一个 2×3 网格子图，每行的 3 个子图使用相同的 y 轴坐标，每列的 2 个子图
使用相同的 x 轴坐标（如图 4-63 所示）：
In[6]: fig, ax = plt.subplots(2, 3, sharex='col', sharey='row')
图 4-63：plt.subplots() 方法共享 x 轴与 y 轴坐标
设置 sharex 与 sharey 参数之后，我们就可以自动去掉网格内部子图的标签，让图形看起
来更整洁。坐标轴实例网格的返回结果是一个 NumPy 数组，这样就可以通过标准的数组
取值方式轻松获取想要的坐标轴了（如图 4-64 所示）：
In[7]: # 坐标轴存放在一个NumPy数组中，按照[row, col]取值
 for i in range(2):
 for j in range(3):
 ax[i, j].text(0.5, 0.5, str((i, j)),
 fontsize=18, ha='center')
 fig
234 ｜ 第 4 章
图 4-64：确定网格中的子图
与 plt.subplot()1 相比，plt.subplots() 与 Python 索引从 0 开始的习惯保持一致。
4.10.4 plt.GridSpec：实现更复杂的排列方式
如果想实现不规则的多行多列子图网格，plt.GridSpec() 是最好的工具。plt.GridSpec()
对象本身不能直接创建一个图形，它只是 plt.subplot() 命令可以识别的简易接口。例如，
一个带行列间距的 2×3 网格的配置代码如下所示：
In[8]: grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)
可以通过类似 Python 切片的语法设置子图的位置和扩展尺寸（如图 4-65 所示）：
In[9]: plt.subplot(grid[0, 0])
 plt.subplot(grid[0, 1:])
 plt.subplot(grid[1, :2])
 plt.subplot(grid[1, 2]);
图 4-65：用 plt.GridSpec 生成不规则子图
这种灵活的网格排列方式用途十分广泛，我经常会用它来创建如图 4-66 所示的多轴频次直
方图（multi-axes histogram）（如图 4-66 所示）：
注 1：与 MATLAB 的索引从 1 开始类似。——译者注
Matplotlib数据可视化 ｜ 235
In[10]: # 创建一些正态分布数据
 mean = [0, 0]
 cov = [[1, 1], [1, 2]]
 x, y = np.random.multivariate_normal(mean, cov, 3000).T
 # 设置坐标轴和网格配置方式
 fig = plt.figure(figsize=(6, 6))
 grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)
 main_ax = fig.add_subplot(grid[:-1, 1:])
 y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)
 x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)
 # 主坐标轴画散点图
 main_ax.plot(x, y, 'ok', markersize=3, alpha=0.2)
 # 次坐标轴画频次直方图
 x_hist.hist(x, 40, histtype='stepfilled',
 orientation='vertical', color='gray')
 x_hist.invert_yaxis()
 y_hist.hist(y, 40, histtype='stepfilled',
 orientation='horizontal', color='gray')
 y_hist.invert_xaxis()
图 4-66：用 plt.GridSpec 可视化多维分布数据
这种类型的分布图十分常见，Seaborn 程序包提供了专门的 API 来实现它们，详情请参见
4.16 节。

## 文字与注释
一个优秀的可视化作品就是给读者讲一个精彩的故事。虽然在一些场景中，这个故事可以
完全通过视觉来表达，不需要任何多余的文字。但在另外一些场景中，辅之以少量的文字
提示（textual cue）和标签是必不可少的。虽然最基本的注释（annotation）类型可能只是
坐标轴标题与图标题，但注释可远远不止这些。让我们可视化一些数据，看看如何通过添
加注释来更恰当地表达信息。还是先在 Notebook 里面导入画图需要用到的一些函数：
```python
In[1]: %matplotlib inline
 import matplotlib.pyplot as plt
 import matplotlib as mpl
 plt.style.use('seaborn-whitegrid')
 import numpy as np
 import pandas as pd
```
让我们用 3.10.4 节介绍过的的数据来演示。在那个案例中，我们画了一幅图表示美国每
一年的出生人数。和前面一样，数据可以在 https://raw.githubusercontent.com/jakevdp/dataCDCbirths/master/births.csv 下载。
首先用前面介绍过的清洗方法处理数据，然后画出结果：
```python
In[2]:
births = pd.read_csv('births.csv')
quartiles = np.percentile(births['births'], [25, 50, 75])
mu, sig = quartiles[1], 0.74 * (quartiles[2] - quartiles[0])
births = births.query('(births > @mu - 5 * @sig) & (births < @mu + 5 * @sig)')
births['day'] = births['day'].astype(int)
births.index = pd.to_datetime(10000 * births.year +
 100 * births.month +
 births.day, format='%Y%m%d')
births_by_date = births.pivot_table('births',
 [births.index.month, births.index.day])
births_by_date.index = [pd.datetime(2012, month, day)
 for (month, day) in births_by_date.index]
In[3]: fig, ax = plt.subplots(figsize=(12, 4))
 births_by_date.plot(ax=ax);
```
在用这样的图表达观点时，如果可以在图中增加一些注释，就更能吸引读者的注意了。可
以通过 plt.text/ ax.text 命令手动添加注释，它们可以在具体的 x / y 坐标点上放上文字：
```python
In[4]: fig, ax = plt.subplots(figsize=(12, 4))
 births_by_date.plot(ax=ax)
 # 在图上增加文字标签
 style = dict(size=10, color='gray')
 ax.text('2012-1-1', 3950, "New Year's Day", **style)
 ax.text('2012-7-4', 4250, "Independence Day", ha='center', **style)
 ax.text('2012-9-4', 4850, "Labor Day", ha='center', **style)
 ax.text('2012-10-31', 4600, "Halloween", ha='right', **style)
 ax.text('2012-11-25', 4450, "Thanksgiving", ha='center', **style)
 ax.text('2012-12-25', 3850, "Christmas ", ha='right', **style)
 # 设置坐标轴标题
 ax.set(title='USA births by day of year (1969-1988)',
 ylabel='average daily births')
 # 设置x轴刻度值，让月份居中显示
 ax.xaxis.set_major_locator(mpl.dates.MonthLocator())
 ax.xaxis.set_minor_locator(mpl.dates.MonthLocator(bymonthday=15))
 ax.xaxis.set_major_formatter(plt.NullFormatter())
 ax.xaxis.set_minor_formatter(mpl.dates.DateFormatter('%h'));
```
'ax.text' 方法需要一个 x 轴坐标、一个 y 轴坐标、一个字符串和一些可选参数，比如文字
的颜色、字号、风格、对齐方式以及其他文字属性。这里用了 ha='right' 与 ha='center'，
ha 是水平对齐方式（horizonal alignment）的缩写。关于配置参数的更多信息，请参考
plt.text() 与 mpl.text.Text() 的程序文档。
### 坐标变换与文字位置
前面的示例将文字放在了目标数据的位置上。但有时候可能需要将文字放在与数据无关的位置
上，比如坐标轴或者图形中。在 Matplotlib 中，我们通过调整坐标变换（transform）来实现。
任何图形显示框架都需要一些变换坐标系的机制。

例如，当一个位于 (x, y) = (1, 1) 位置的
点需要以某种方式显示在图上特定的位置时，就需要用屏幕的像素来表示。用数学方法处
理这种坐标系变换很简单，Matplotlib 有一组非常棒的工具可以实现类似功能（这些工具
位于 matplotlib.transforms 子模块中）。
虽然一般用户并不需要关心这些变换的细节，但是了解这些知识对在图上放置文字大有帮
助。一共有三种解决这类问题的预定义变换方式。  
'ax.transData'
以数据为基准的坐标变换。
'ax.transAxes'
以坐标轴为基准的坐标变换（以坐标轴维度为单位）。
'fig.transFigure'
以图形为基准的坐标变换（以图形维度为单位）。
下面举一个例子，用三种变换方式将文字画在不同的位置：
```python
In[5]: fig, ax = plt.subplots(facecolor='lightgray')
 ax.axis([0, 10, 0, 10])
 # 虽然transform=ax.transData是默认值，但还是设置一下
 ax.text(1, 5, ". Data: (1, 5)", transform=ax.transData)
 ax.text(0.5, 0.1, ". Axes: (0.5, 0.1)", transform=ax.transAxes)
 ax.text(0.2, 0.2, ". Figure: (0.2, 0.2)", transform=fig.transFigure);
```
默认情况下，上面的文字在各自的坐标系中都是左对齐的。这三个字符串开头的 . 字符基
本就是对应的坐标位置。
transData 坐标用 x 轴与 y 轴的标签作为数据坐标。transAxes 坐标以坐标轴（图中白色矩
形）左下角的位置为原点，按坐标轴尺寸的比例呈现坐标。transFigure 坐标与之类似，
不过是以图形（图中灰色矩形）左下角的位置为原点，按图形尺寸的比例呈现坐标。
需要注意的是，假如你改变了坐标轴上下限，那么只有 transData 坐标会受影响，其他坐
标系都不变：
```python
In[6]: ax.set_xlim(0, 2)
 ax.set_ylim(-6, 6)
 fig
```
如果你改变了坐标轴上下限，那么就可以更清晰地看到刚刚所说的变化。如果你是在
Notebook 里运行本书代码的，那么可以把 %matplotlib inline 改成 %matplotlib notebook，
然后用图形菜单与图形交互（拖动按钮即可），就可以实现坐标轴平移了。
### 箭头与注释
除了刻度线和文字，简单的箭头也是一种有用的注释标签。
在 Matplotlib 里面画箭头通常比你想象的要困难。虽然有一个 plt.arrow() 函数可以实现这
个功能，但是我不推荐使用它，因为它创建出的箭头是 SVG 向量图对象，会随着图形分
辨率的变化而改变，最终的结果可能完全不是用户想要的。我要推荐的是 plt.annotate()
函数。这个函数既可以创建文字，也可以创建箭头，而且它创建的箭头能够进行非常灵活
的配置。
下面用 annotate 的一些配置选项来演示：
```python
In[7]: %matplotlib inline
 fig, ax = plt.subplots()
 x = np.linspace(0, 20, 1000)
 ax.plot(x, np.cos(x))
 ax.axis('equal')
 ax.annotate('local maximum', xy=(6.28, 1), xytext=(10, 4),
 arrowprops=dict(facecolor='black', shrink=0.05))
 ax.annotate('local minimum', xy=(5 * np.pi, -1), xytext=(2, -6),
 arrowprops=dict(arrowstyle="->",
 connectionstyle="angle3,angleA=0,angleB=-90"));
```
箭头的风格是通过 arrowprops 字典控制的，里面有许多可用的选项。由于这些选项在
Matplotlib 的官方文档中都有非常详细的介绍，我就不再赘述，仅做一点儿功能演示。让
我们用前面的美国出生人数图来演示一些箭头注释：
```python
In[8]:
fig, ax = plt.subplots(figsize=(12, 4))
births_by_date.plot(ax=ax)
# 在图上增加箭头标签
ax.annotate("New Year's Day", xy=('2012-1-1', 4100), xycoords='data',
 xytext=(50, -30), textcoords='offset points',
 arrowprops=dict(arrowstyle="->",
 connectionstyle="arc3,rad=-0.2"))
ax.annotate("Independence Day", xy=('2012-7-4', 4250), xycoords='data',
 bbox=dict(boxstyle="round", fc="none", ec="gray"),
 xytext=(10, -40), textcoords='offset points', ha='center',
 arrowprops=dict(arrowstyle="->"))
ax.annotate('Labor Day', xy=('2012-9-4', 4850), xycoords='data', ha='center',
 xytext=(0, -20), textcoords='offset points')
ax.annotate('', xy=('2012-9-1', 4850), xytext=('2012-9-7', 4850),
 xycoords='data', textcoords='data',
 arrowprops={'arrowstyle': '|-|,widthA=0.2,widthB=0.2', })
ax.annotate('Halloween', xy=('2012-10-31', 4600), xycoords='data',
 xytext=(-80, -40), textcoords='offset points',
 arrowprops=dict(arrowstyle="fancy",
 fc="0.6", ec="none",
 connectionstyle="angle3,angleA=0,angleB=-90"))
ax.annotate('Thanksgiving', xy=('2012-11-25', 4500), xycoords='data',
 xytext=(-120, -60), textcoords='offset points',
 bbox=dict(boxstyle="round4,pad=.5", fc="0.9"),
 arrowprops=dict(arrowstyle="->",
 connectionstyle="angle,angleA=0,angleB=80,rad=20")) 

ax.annotate('Christmas', xy=('2012-12-25', 3850), xycoords='data',
 xytext=(-30, 0), textcoords='offset points',
 size=13, ha='right', va="center",
 bbox=dict(boxstyle="round", alpha=0.1),
 arrowprops=dict(arrowstyle="wedge,tail_width=0.5", alpha=0.1));
# 设置坐标轴标题
ax.set(title='USA births by day of year (1969-1988)',
 ylabel='average daily births')
# 设置x轴刻度值，让月份居中显示
ax.xaxis.set_major_locator(mpl.dates.MonthLocator())
ax.xaxis.set_minor_locator(mpl.dates.MonthLocator(bymonthday=15))
ax.xaxis.set_major_formatter(plt.NullFormatter())
ax.xaxis.set_minor_formatter(mpl.dates.DateFormatter('%h'));
ax.set_ylim(3600, 5400);
```
你可能已经注意到了，箭头和文本框的配置功能非常细致，这样你就可以创建自己想要的
箭头风格了。不过，功能太过细致往往也就意味着操作起来比较复杂，如果真要做一个产
品级的图形，可能得耗费大量的时间。最后我想说一句，前面适用的混合风格并不是数据
可视化的最佳实践，仅仅是为演示一些功能而已。
关于箭头和注释风格的更多介绍与示例，可以在 Matplotlib 的画廊（gallery）中看到，尤
其推荐 http://matplotlib.org/examples/pylab_examples/annotation_demo2.html 这个例子。

## 自定义坐标轴刻度
虽然 Matplotlib 默认的坐标轴定位器（locator）与格式生成器（formatter）可以满足大部分
需求，但是并非对每一幅图都合适。本节将通过一些示例演示如何将坐标轴刻度调整为你
需要的位置与格式。
在介绍示例之前，我们最好先对 Matplotlib 图形的对象层级有更深入的理解。Matplotlib 的
目标是用 Python 对象表现任意图形元素。例如，想想前面介绍的 figure 对象，它其实就
是一个盛放图形元素的包围盒（bounding box）。可以将每个 Matplotlib 对象都看成是子对
象（sub-object）的容器，例如每个 figure 都会包含一个或多个 axes 对象，每个 axes 对象
又会包含其他表示图形内容的对象。
坐标轴刻度线也不例外。每个 axes 都有 xaxis 和 yaxis 属性，每个属性同样包含构成坐标
轴的线条、刻度和标签的全部属性。
### 主要刻度与次要刻度
每一个坐标轴都有主要刻度线与次要刻度线。顾名思义，主要刻度往往更大或更显著，而
次要刻度往往更小。虽然一般情况下 Matplotlib 不会使用次要刻度，但是你会在对数图中
看到它们：
```python
In[1]: %matplotlib inline
 import matplotlib.pyplot as plt
 plt.style.use('seaborn-whitegrid')
 import numpy as np
In[2]: ax = plt.axes(xscale='log', yscale='log')
```
我们发现每个主要刻度都显示为一个较大的刻度线和标签，而次要刻度都显示为一个较小
的刻度线，且不显示标签。
可以通过设置每个坐标轴的 formatter 与 locator 对象，自定义这些刻度属性（包括刻度
线的位置和标签）。来检查一下图形 x 轴的属性：
```python
In[3]: print(ax.xaxis.get_major_locator())
 print(ax.xaxis.get_minor_locator())
<matplotlib.ticker.LogLocator object at 0x107530cc0>
<matplotlib.ticker.LogLocator object at 0x107530198>
In[4]: print(ax.xaxis.get_major_formatter())
 print(ax.xaxis.get_minor_formatter())
<matplotlib.ticker.LogFormatterMathtext object at 0x107512780>
<matplotlib.ticker.NullFormatter object at 0x10752dc18>
```
我们会发现，主要刻度标签和次要刻度标签的位置都是通过一个 LogLocator 对象（在对数
图中可以看到）设置的。然而，次要刻度有一个 NullFormatter 对象处理标签，这样标签
就不会在图上显示了。
下面来演示一些示例，看看不同图形的定位器与格式生成器是如何设置的。
### 隐藏刻度与标签
最常用的刻度 / 标签格式化操作可能就是隐藏刻度与标签了，可以通过 plt.NullLocator()
与 plt.NullFormatter() 实现，如下所示：
```python
In[5]: ax = plt.axes()
 ax.plot(np.random.rand(50))
 ax.yaxis.set_major_locator(plt.NullLocator())
 ax.xaxis.set_major_formatter(plt.NullFormatter())
```
需要注意的是，我们移除了 x 轴的标签（但是保留了刻度线 / 网格线），以及 y 轴的刻度
（标签也一并被移除）。在许多场景中都不需要刻度线，比如当你想要显示一组图形时。举
个例子，像图 4-75 那样包含不同人脸的照片，就是经常用于研究有监督机器学习问题的示
例（详情请参见 5.7 节）：
```python
In[6]: fig, ax = plt.subplots(5, 5, figsize=(5, 5))
 fig.subplots_adjust(hspace=0, wspace=0)
 # 从scikit-learn获取一些人脸照片数据
 from sklearn.datasets import fetch_olivetti_faces
 faces = fetch_olivetti_faces().images
 for i in range(5):
 for j in range(5):
 ax[i, j].xaxis.set_major_locator(plt.NullLocator())
 ax[i, j].yaxis.set_major_locator(plt.NullLocator())
 ax[i, j].imshow(faces[10 * i + j], cmap="bone")
```
需要注意的是，由于每幅人脸图形默认都有各自的坐标轴，然而在这个特殊的可视化场景
中，刻度值（本例中是像素值）的存在并不能传达任何有用的信息，因此需要将定位器设
置为空。
### 增减刻度数量
默认刻度标签有一个问题，就是显示较小图形时，通常刻度显得十分拥挤。我们可以在图  
'In[7]: fig, ax = plt.subplots(4, 4, sharex=True, sharey=True)'  

尤其是 x 轴，数字几乎都重叠在一起，辨识起来非常困难。我们可以用 plt.MaxNLocator()
来解决这个问题，通过它可以设置最多需要显示多少刻度。根据设置的最多刻度数量，
Matplotlib 会自动为刻度安排恰当的位置：
```python
In[8]: # 为每个坐标轴设置主要刻度定位器
 for axi in ax.flat:
 axi.xaxis.set_major_locator(plt.MaxNLocator(3))
 axi.yaxis.set_major_locator(plt.MaxNLocator(3))
 fig
```
这样图形就显得更简洁了。如果你还想要获得更多的配置功能，那么可以试试 plt.
MultipleLocator，我们将在接下来的内容中介绍它。
### 花哨的刻度格式
Matplotlib 默认的刻度格式可以满足大部分的需求。虽然默认配置已经很不错了，但是有
时候你可能需要更多的功能:
```python
In[9]: # 画正弦曲线和余弦曲线
 fig, ax = plt.subplots()
 x = np.linspace(0, 3 * np.pi, 1000)
 ax.plot(x, np.sin(x), lw=3, label='Sine')
 ax.plot(x, np.cos(x), lw=3, label='Cosine')
 # 设置网格、图例和坐标轴上下限
 ax.grid(True)
 ax.legend(frameon=False)
 ax.axis('equal')
 ax.set_xlim(0, 3 * np.pi);
```
我们可能想稍稍改变一下这幅图。首先，如果将刻度与网格线画在 π 的倍数上，图形
会更加自然。可以通过设置一个 MultipleLocator 来实现，它可以将刻度放在你提供的

数值的倍数上。为了更好地测量，在 π/4 的倍数上添加主要刻度和次要刻度
```python
In[10]: ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
 ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 4))
 fig
```
然而，这些刻度标签看起来有点奇怪：虽然我们知道它们是 π 的倍数，但是用小数表示圆
周率不太直观。因此，我们可以用刻度格式生成器来修改。由于没有内置的格式生成器可
以直接解决问题，因此需要用 plt.FuncFormatter 来实现，用一个自定义的函数设置不同
刻度标签的显示：
```python
In[11]: def format_func(value, tick_number):
 # 找到π/2的倍数刻度
 N = int(np.round(2 * value / np.pi))
 if N == 0:
 return "0"
 elif N == 1:
 return r"$\pi/2$"
 elif N == 2:
 return r"$\pi$"
 elif N % 2 > 0:
 return r"${0}\pi/2$".format(N)
 else:
 return r"${0}\pi$".format(N // 2)
 ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
 fig
 ```
这样就好看多啦！其实我们已经用了 Matplotlib 支持 LaTeX 的功能，在数学表达式两侧加
上美元符号（$），这样可以非常方便地显示数学符号和数学公式。在这个示例中，"$\pi$"
就表示圆周率符合 π。
当你准备展示或打印图形时，plt.FuncFormatter() 不仅可以为自定义图形刻度提供十分灵
活的功能，而且用法非常简单。

### 格式生成器与定位器小结
前面已经介绍了一些格式生成器与定位器，下面用表格简单地总结一下内置的格式生成器
与定位器选项。关于两者更详细的信息，请参考各自的程序文档或者 Matplotlib 的在线文
档。以下的所有类都在 plt 命名空间内。
定位器类 描述
NullLocator 无刻度
FixedLocator 刻度位置固定
IndexLocator 用索引作为定位器（如 x = range(len(y)) ）
LinearLocator 从 min 到 max 均匀分布刻度
LogLocator 从 min 到 max 按对数分布刻度
MultipleLocator 刻度和范围都是基数（base）的倍数
MaxNLocator 为最大刻度找到最优位置
AutoLocator （默认）以 MaxNLocator 进行简单配置
AutoMinorLocator 次要刻度的定位器
格式生成器类 描述
NullFormatter 刻度上无标签
IndexFormatter 将一组标签设置为字符串
FixedFormatter 手动为刻度设置标签
FuncFormatter 用自定义函数设置标签
FormatStrFormatter 为每个刻度值设置字符串格式
ScalarFormatter （默认）为标量值设置标签
LogFormatter 对数坐标轴的默认格式生成器
我们将在后面的章节中看到使用这些功能的更多示例。
