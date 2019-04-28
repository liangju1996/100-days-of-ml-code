# 学习日志
## 2019.4.28
想系统机器学习，发现python功底不够。于是转战学习python编程：从入门到实践

## 2019.4.25
学习K近邻代码

## 2019.4.19-2019.22
### 学习逻辑回归及其代码
这两天忙着改论文，没大学习........

## 2019.4.18
### 学习day4 逻辑回归
### 遇到md公式编辑问题，尚未编辑成功,抽空再改
[参考1](https://blog.csdn.net/lihaoweicsdn/article/details/83895143)  
[参考2](https://blog.csdn.net/wireless_com/article/details/70596155)  

## 2019.4.17
### 在GitHub仓库中建立一个文件夹
* 法一：在电脑上创建好，直接拖拽电脑上（像上传图片一样)
* 法二：点击Create new file 在文件名后面加上“/”+回车+文件名，在里面写点东西+提交。  
eg.输入ab/cd--->产生一个名为ab的文件夹，里面有个名为cd的文件
### 学习day3 多元线性回归
* toarray(): 作用-->独热码以稀疏矩阵的方式存储，跟array不一样，所以需要用toarray转化
onehot会增加几个个虚拟变量，里面好多0导致矩阵使用稀疏矩阵存储，存储方式为(x,y) value的格式，(x,y)表示坐标，value表示值。
稀疏矩阵： 零特别多，，，存储时只存非零值及其坐标

* array(): 作用，将list等转化为一维数组
* 当有一个函数或者方法的作用无法确定时  
可以分别加上和去掉它，查看两者结果的区别  
eg.查看toarray（）的作用：  
X = onehotencoder.fit_transform(X)
print('之前为：')
print(X)
after = X.toarray()
print('之后为：')
print(after)
### 讨论所得
* read_cvs()-->将要读取的文件放在与代码相同的目录中则可以直接读，不用加路径
* [为什么引入独热码，使用独热码时的计算原理](https://zhuanlan.zhihu.com/p/39012149)
* [虚拟变量及其设置原则](https://wenku.baidu.com/view/7265e32126284b73f242336c1eb91a37f11132f9)
* [虚拟变量陷阱的原理](https://www.jianshu.com/p/1ff8aa30ec64)如果不消除虚拟变量陷阱的话会产生完全共线性（列相关），原因是除了n个变量之外还有一个截距，使得该矩阵为(n,n+1) 是一个非列满秩矩阵
* [不产生虚拟变量陷阱的情况：去掉截距项](https://bbs.pinggu.org/thread-1273924-1-1.html)
* [虚拟变量陷阱的消除](https://wenku.baidu.com/view/7265e32126284b73f242336c1eb91a37f11132f9)去掉任意一列（冗余项）-->消除之后依然可以表示n个状态.但是有一项为零，其他项都是1使得他们大小（欧式距离）不同了，这样依然可以（因为有个截距的问题，比如这去除第一个量之后的俩变量x1,x2，我们的线性回归模型是y=a*x1+b*x2+d，因为d的存在使得哪怕x1与x2都为0，预测值依然不会出问题）

## 2019.4.16
### 使用saver.restore()时一直出错
* 解决方法：  
将之前的  
```saver.restore(sess, './model.ckpt') ```  
改为  
```saver.save(sess, os.path.join(os.getcwd(), 'model.ckpt'))```
### 搭建tensorflow-gpu环境
### 学习day2 一元线性回归

## 2019.4.15
### 复习完善day 1 内容
* 将自己的理解放在注释中，发现文档注释在md文件中的显示与代码相似
### 遇到不懂的函数、模块等有以下解决方法：
* 查阅官网
* 右击`Go to Definition`
* `F12`

`发现自己的不足`:效率极低！

## 2019.4.14
### 安装anaconda3
安装成功，未出现问题，但...超级慢  据说使用清华大学的镜像网站会快一点


## 2019.4.13
### 学习md文件的编辑方法  
不足之处参考md教程下方链接`更详细`
* 遇到的问题：
  * 表格始终无法显示
* 多次尝试后，发现表格的关键在于第二行，不使用冒号时`|`两端必须！！！有空格,  
表格与其前一行中间必须有空行。
>补充：
* 今天看到[yunhao1996发的链接](https://blog.csdn.net/Cassie_zkq/article/details/79968598),意识到自己好像没有注意图片的问题.  
![桔子](https://github.com/liangju1996/100-days-of-ml-code/blob/master/timg.jpg)










