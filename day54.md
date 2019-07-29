# 层次聚类
### 什么是层次聚类？
层次聚类，顾名思义，是一种构建簇的层次结构的算法。  
该算法从分配给自己簇的所有数据点开始。  
然后两个距离最近的簇合并为同一个簇。  
最后，当只剩下一个簇时，该算法终止。  
有两种类型的层次聚类：分裂和凝聚。  
![层次聚类](https://github.com/liangju1996/100-days-of-ml-code/blob/master/图片/层次聚类1.png)
### 凝聚层次聚类
这里的每个对象最初被认为是单一的簇（叶子结点）。
然后，最相似的簇依次合并，直到形成了一个大的簇（根节点）。
簇的层次结构表示为树（或树状图）。  

![凝聚层次聚类](https://github.com/liangju1996/100-days-of-ml-code/blob/master/图片/凝聚层次聚类.png)

### 树状图
书的根结点是唯一的簇，它囊括了所有的样本，而叶子结点是只有一个样本的簇。

![原理](https://github.com/liangju1996/100-days-of-ml-code/blob/master/图片/原理.png)


通过观察树状图，可以很好的判断出不同组的簇数。  
水平线贯穿过的树状图中垂直线的数量将是簇数的最佳选择，  
这条线保证了垂直横穿的最大距离并且不与簇相交。

## 代码
```python
# 导入库
import numpy as np
import matplotlib.pyplot as plt  #用于画图工具
from sklearn.cluster import AgglomerativeClustering #层次聚类
from sklearn.neighbors import kneighbors_graph #最邻近搜索
# 定义一个实现凝聚层次聚类的函数
def perform_clustering(X, connectivity, title, num_clusters=3, linkage='ward'):
    plt.figure()
        # 定义凝聚层次聚类模型
    model = AgglomerativeClustering(linkage=linkage, connectivity=connectivity, n_clusters=num_clusters)
    model.fit(X)  # 训练模型
# 提取标记，然后指定不同聚类在图形中的标记：
    labels = model.labels_  # 提取标记
    markers = '.vx'  # 为每种集群设置不同的标记
# 迭代数据，用不同的标记把聚类的点画在图形中
    for i, marker in zip(range(num_clusters), markers):
        # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，
        # 然后返回由这些元组组成的对象，这样做的好处是节约了不少的内存。
    # 画出属于某个集群中心的数据点
        plt.scatter(X[labels==i, 0], X[labels==i, 1], s=50,marker=marker, color='k', facecolors='none')
        plt.title(title)

# 为了演示凝聚层次聚类的优势，我们用它对一些在空间中是连接在一起、但彼此却非常
# 接近的数据进行聚类。我们希望连接在一起的数据可以聚成一类，而不是在空间上非常接近的点
# 聚成一类。下面定义一个函数来获取一组呈螺旋状的数据点：
# 定义函数获取螺旋状的数据点
def get_spiral(t, noise_amplitude=0.5):
    r = t
    x = r * np.cos(t)
    y = r * np.sin(t)
    return add_noise(x, y, noise_amplitude)
# 在上面的函数中，我们增加了一些噪声，因为这样做可以增加一些不确定性。下面定义
# 噪声函数：
def add_noise(x, y, amplitude):
    X = np.concatenate((x, y))
    X += amplitude * np.random.randn(2, X.shape[1])
    return X.T
# 我们再定义一个函数来获取位于玫瑰曲线上的数据点（rose curve，又称为rhodonea curve，
# 极坐标中的正弦曲线）：
def get_rose(t, noise_amplitude=0.02):
    # 设置玫瑰曲线方程；如果变量k是奇数，那么曲线有k朵花瓣；如果k是偶数，那么有2k朵花瓣
    k = 5
    r = np.cos(k*t) + 0.25
    x = r * np.cos(t)
    y = r * np.sin(t)
    return add_noise(x, y, noise_amplitude)
# 为了增加多样性，我们再定义一个hypotrochoid函数：
def get_hypotrochoid(t, noise_amplitude=0):
    a, b, h = 10.0, 2.0, 4.0
    x = (a - b) * np.cos(t) + h * np.cos((a - b) / b * t)
    y = (a - b) * np.sin(t) - h * np.sin((a - b) / b * t)
    return add_noise(x, y, 0)

if __name__=='__main__':
    # 如果我们是直接执行某个.py文件的时候，该文件中那么”__name__ == '__main__'
    # “是True,但是我们如果从另外一个.py文件通过import导入该文件的时候，这时__name__的值就是我们这个py文件的名字而不是__main__。
    #
    # 这个功能还有一个用处：调试代码的时候，在”if __name__ == '__main__'“中加入一些我们的调试代码，
    # 我们可以让外部模块调用的时候不执行我们的调试代码，但是如果我们想排查问题的时候，直接执行该模块文件，调试代码能够正常运行！
    # 生成样本数据
    n_samples = 500
    np.random.seed(2)
    t = 2.5 * np.pi * (1 + 2 * np.random.rand(1, n_samples))
    X = get_spiral(t)
    # 不考虑螺旋形的数据连接性
    connectivity = None
    perform_clustering(X, connectivity, 'No connectivity')
    # 根据数据连接线创建K个临近点的图形
    connectivity = kneighbors_graph(X, 10, include_self=False)
    perform_clustering(X, connectivity, 'K-Neighbors connectivity')
    plt.show()











```

