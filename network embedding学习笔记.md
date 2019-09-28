# network embedding学习笔记

network，是指例如社交应用中用户之间的好友关系形成的一张网，或者博客应用中用户、博客之间的关注、订阅关系，或者web网页相互之间的link诸如此类的等等网络形态

network embedding，类似nlp中的词向量，就是要对network中的实体（节点或者边）用向量表示，向量之间的距离表示实体之间的相似度。是网络中分类、聚类、推荐等应用的基础设施。

网友有一张图很好的展现了他们之间的关系：

![](img/network_embedding/ne.png)

该图引用自下面的文章，该文章也是ne方面很全面的信息索引。

```
https://github.com/chihming/awesome-network-embedding #论文与实现的索引
http://networkx.github.io/  # 网络操作python库
http://socialcomputing.asu.edu/pages/datasets # 公开的数据集
```

## 1、DeepWalk算法

DeepWalk算法思想比较简单直观：在网络中随机游走，将路径记录下来，每条路径看做一个句子，经过的节点看做是句子中的词，然后使用skip gram算法训练词向量。

[python示例代码在这里](https://github.com/bisonliao/daydayup/blob/master/mxnet/networkEmbedding_DeepWalk.py)

由于公开数据集普遍较大，训练出来的NE不容易看到直观的效果，我自己定义一个小小的网络，通过聚类和图形展示来验证其效果。

网络结构如下：

![](img/network_embedding/small_network.jpg)

训练维度为2的embedding节点向量，使用kmeans聚类（需要指定簇的个数，3），输出的各节点的标签为：

```
[1 1 1 1 1 1 2 2 2 2 0 0 0 0]
```

可以看出，聚类为3个，前6个节点为一个簇，中间四个节点为一个簇，后4个节点为一个簇，符合预期。

使用DBScan算法聚类，可以不指定簇的个数，需要微调最大距离等参数，也可以做到符合预期的聚类：

```python
cl = DBSCAN(min_samples=3, metric='cosine', eps=0.05)
print(cl.fit_predict(m))
```

可视化显示出来如下：

![](img/network_embedding/position.jpg)

节点2的最近5个点是3、1、4、6、5，前面的小数是余弦相似度：

```
[
 (0.9994833469390869, 'u3'), 
 (0.9992480874061584, 'u1'), 
 (0.9983071684837341, 'u4'), 
 (0.9981566071510315, 'u6'), 
 (0.9899411797523499, 'u5')
 ]
```

对1万多个节点的BlogCatalog进行embedding，与官方的embedding结果对比，同一个节点（#58）的top5相似节点集合不一致，交集为0。与类似节点的相邻节点集合比较jaccard相似度，两个实现的jaccard相似度接近。

## 2、node2vec算法

node2vec算法类似DeepWalk算法，通过两个参数p、q来控制游走过程中的策略：

![](img/network_embedding/node2vec.jpg)

没有太能理解p、q对相似性的影响的理解。下面是P、Q两种典型情况：

![](img/network_embedding/node2vec_2.jpg)

[我自己的python示例代码在这里](https://github.com/bisonliao/daydayup/blob/master/mxnet/networkEmbedding_Node2Vec.py)

pip3可以安装一个叫做node2vec的包，它使用gensim.models.word2vec.Word2Vec来训练词向量。官网在：

```
https://github.com/eliorc/node2vec
```

对比效果如下：

![](img/network_embedding/node2vec_3.jpg)

我自己的代码对13000多个节点的实际网络进行embedding（P和Q等于1），抽查聚类后的簇内的边的密度和簇间的边的密度，发现差异很小，簇内和簇间的平均余弦距离也没有明显差异，不太符合预期，说明embedding效果不好：

```
c1, c2 size:186,72
17205 avg cos distances in cluster:0.72
13392 avg cos distances between cluster:0.77
17205 avg edges dense in cluster:0.01267
13392 avg edges dense between cluster:0.00717
```

node2vec包的embedding后的聚类效果也不比我的代码的效果好：

```
c1, c2 size:111,192
6105 avg cos distances in cluster:0.75
21312 avg cos distances between cluster:0.81
6105 avg edges dense in cluster:0.00426
21312 avg edges dense between cluster:0.00511
```

节点2的相似节点，两者给出的答案也不一致：

```
the similar node of #2: 3400 2241 1739 4123 4007 2090 1509 4209 407 8061
the similar node of #2: 7389 3017 3050 3400 3345 7578 4372 7589 7113 4780 
```

[pip3 node2vec包调用的代码在这里](https://github.com/bisonliao/daydayup/blob/master/mxnet/networkEmbedding_Node2Vec_official.py)

## 3、基于图的因子分解的算法

基本思想是：将邻接矩阵 Y 分解为 U 点乘 U的转置的形式。U的每一行就是一个节点的embedding。

假设网络中有N个节点，那么Y的shape是(N, N)， network embedding的维度假设是r，那么U的shape是    （N，r），U的转置的shape是（r，N）。

已知Y，采用梯度下降的方法可以求得U。损失函数是：
$$
1/2*|| Y - U.Transpose(U)||^2 + λ/2 * ||Y||^2
$$
后面这项是正则化部分。

一般而言，邻接矩阵 对角线上的元素可能为0，但U和U的转置的点积结果的对角线元素会大于0，这里需要特殊处理一下。

另外需要注意的是，r不能太小，否则无法比较彻底的收敛，因为自由度不够。从方程组求解的角度也比较好理解，U里面有 r x N个未知数，而Y中每个元素都对应一个方程，即有N x N个约束。对于无向图，邻接矩阵是对称矩阵，即 Y = Transpose(Y)，有N(N-1)/2 个约束，所以 r 不能小于(N-1)/2

上面的14个节点的小网络的例子，r 取8的时候，损失函数可以比较好的收敛，点乘的结果非常接近Y，embedding结果符合预期：

```
ep:9900, loss:0.0106
聚类标签：[0 0 0 0 0 0 2 2 2 2 1 1 1 1]
节点#2的相似节点：[ 3, 6, 1, 5, 4, 7]
```

因为我们的目的不是要求的U完全满足点乘后的结果等于Y，而是求得对节点的embedding，所以不要求彻底的收敛。而且当节点数很多的时候，r 也会很大，不切实际。 我们看一下把 r 设置为2的embedding效果怎么样。

可以看到，损失函数收敛程度不怎么好，但embedding效果符合预期：

```
ep:9900, loss:0.1072
聚类的簇标签：[0 0 0 0 0 0 2 2 2 2 1 1 1 1]
节点#2的相似节点：[4, 6, 1, 3, 5, 7]
```

![](img/network_embedding/graph_factorization1.jpg)

对13000多个节点的实际网络进行测试，对embedding进行聚类，对比簇内和簇间的边的密度、簇内和簇间的cosin距离。

可以看到：通过调整epsilon参数，DBScan算法将embedding后的网络节点分为17个簇，另外有1314个节点被认为是噪声点没有归入任何一簇。标签为0的簇特别大，包含了8930个节点。

任意抽查两个簇，簇内的边的密度为0.25，簇之间的边密度为0。

符合预期。

```
#DBScan聚类效果
cluster labels: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, -1}
noisy node: 1314
c1, c2 size:8,6
28 avg cos distances in cluster:0.26
48 avg cos distances between cluster:0.67
28 avg edges dense in cluster:0.25
48 avg edges dense between cluster:0.00
#kmeans聚类效果，指定簇个数为100：
cluster labels: {0, 1, 2, 3,..., 98, 99}
noisy node: 0
c1, c2 size:17,355
136 avg cos distances in cluster:0.61
6035 avg cos distances between cluster:1.23
136 avg edges dense in cluster:0.35
6035 avg edges dense between cluster:0.01
```

[python示例代码在这里](https://github.com/bisonliao/daydayup/blob/master/mxnet/networkEmbedding_GraphFactor.py)

这个算法很明显的优势是：

1. 训练速度快
2. 方便分块进行计算，或者分布式计算