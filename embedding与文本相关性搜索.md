### embedding与文本相关性搜索

#### 1、问题提出

一个技术服务公司的官网通常提供了他们产品的技术文档。如何提供一个好用的搜索？

#### 2、解决方案

用搜索引擎是一种方案，也有开源的可私有化部署的搜索引擎。结合最近很风光的openai公司的能力，我们可以有这样一个思路：

1. 用openai的开放的embedding接口和强大的文本理解能力，把文档转化为1536维大小的向量，并调用openai的接口生成文档的摘要
2. 利用redis/ElasticSearch等向量查询数据库，把文档的url和向量存储起来
3. 当用户输入一个查询语句，把它转化为1536维的向量，并到数据库里进行向量相似性搜索，找到相似的一批向量，把对应文档的url和摘要返回给用户

##### 2.1 数据预处理和embedding

##### 2.2 数据入库

##### 2.3 数据查询

##### 2.4 也可以利用GPU进行暴力搜索

向量相似度搜索（VSS），本质就是要找到向量空间里k个最近的邻居，即KNN问题，该问题有很多算法，例如KDTree、局部敏感hash（LSHash）、K-Means聚类方法。

如果请求量不大、文档不多，也可以直接用GPU暴力穷举的方式查找最近的k个邻居，前提是你得有一块支持CUDA计算框架的GPU。

如何安装CUDA和python相关库不展开，可以看参考文档，这里给出使用python+CUDA的代码示例如下：

```python
#!python3
import numpy as np
import cupy as cp
import datetime
import time

ROW=200000 #不能太大，受限于GPU显存大小，而且还有几个和矩阵等大小的中间运算结果都是要占用显存的
COL=1536

unix_timestamp = (datetime.datetime.now() - datetime.datetime(1970, 1, 1)).total_seconds()
np.random.seed(int(unix_timestamp))
#模拟生产一个矩阵，里面保存了ROW个文档的embedding后的向量，每个向量是由COL个float32型元素组成
def genData()->np.ndarray:
    return np.random.randint(-65535, 65535, (ROW, COL), np.int32) / 65537

#两个核函数，用于余弦相似度计算
# 第一个核函数负责计算两个矩阵逐个元素的乘积，结果为同样shape的元素
multiply = cp.ElementwiseKernel(
   'float32 x, float32 y',
   'float32 z',
   'z = x * y',
   'multiply')
# 第二个核函数负责计算三个矩阵逐个元素的 x / (sqrt(y) * sqrt(z))
similar = cp.ElementwiseKernel(
   'float32 x, float32 y, float32 z',
   'float32 result',
   'result = x / (sqrt(y) * sqrt(z))',
   'similar')

a = genData()# type:np.ndarray
a = a.astype(np.float32)
a = cp.asarray(a) #传输到GPU设备上去

'''b = genData() #type:np.ndarray
b = b.astype(np.float32)
b = cp.asarray(b)'''
# 假设要搜索的元素的embedding等于a[1]，把它broadcast为同a的形状，a和b可以逐行求余弦相似性
b = cp.broadcast_to(a[1], (ROW,COL))

print("begin calclulate...")
# 求两个矩阵a,b的逐行的余弦相似性，为了充分利用GPU并行计算的能力，分解为这么几步:
# step1：a和b逐个元素相乘，保存为ab
# step2：a和a逐个元素相乘，保存为aa；b和b逐个元素相乘保存为bb
# step3：ab、aa和bb这三个矩阵逐行加和，得到三个矩阵的形状为(ROW, 1)
# step4: ab aa和bb这三个矩阵逐个元素运行 x/(sqrt(y)*sqrt(z))的运算，得到ROW个相似度，其中最小的K个就对应搜索结果
ab = multiply(a, b) #type:cp.ndarray
aa = multiply(a, a) #type:cp.ndarray
bb = multiply(b, b) #type:cp.ndarray

ab = ab.sum(axis=1)
aa = aa.sum(axis=1)
bb = bb.sum(axis=1)
#time.sleep(10) #这里睡眠10秒是为了通过gpu-z软件观测到占用了多少GPU内存
cc = similar(ab, aa, bb) #type:cp.ndarray

cc = cc.get() # copy to host memory as an np.ndarray
print(cc)
print(cc.max(), cc.min())
```

参考文档：

```
https://developer.nvidia.com/cuda-python
https://docs.cupy.dev/en/stable/install.html
https://docs.cupy.dev/en/stable/user_guide/kernel.html
```

#### 3、使用pretrained模型本地化部署实现embedding也是可以的

```python
from fastparquet import ParquetFile
import pandas as pd
import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer,util # from:https://huggingface.co/sentence-transformers

'''
#下面这样都会报错，我也不知道该怎么搞
pf = ParquetFile("E:\\Temp\\download\\train-00000-of-00001-ebc48879f34571f6.parquet") #type:ParquetFile
df = pf.to_pandas() #type:pd.DataFrame
'''
#把从huggingface下载的新闻语料进行embdding，并保存到本地文件
def embedNews():
    # dataset from https://huggingface.co/datasets/argilla/news-summary
    dataset = pq.ParquetDataset("E:\\Temp\\download\\test-00000-of-00001-6227bd8eb10a9b50.parquet")
    df = dataset.read_pandas().to_pandas() #type:pd.DataFrame

    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embList = list()
    print(df.shape)
    for i in range(df.shape[0]):
        sentence = df.at[i, "text"]
        emb = model.encode(sentence)
        embList.append(emb)
        if (i%97) == 7:
            print(i)
    embDict = {'emb':embList, 'text':df["text"]}
    embDict = pd.DataFrame(embDict) #type:pd.DataFrame
    print(embDict.shape)
    embDict.to_pickle("e:\\news.pkl")

#从本地文件加载embedding向量，然后搜索与sentence相关的
def search(sentence:str):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embdf = pd.read_pickle("e:\\news.pkl") #type:pd.DataFrame
    tensor = torch.tensor(embdf["emb"])
    emb = model.encode(sentence)

    result = util.semantic_search(emb, tensor, top_k=4) #type:list
    #result is a list of list. each entry is a list of dictionary, each dictionary is for a similar text
    # corpus_id and score
    result = result[0] #type:list
    for i in range(len(result)):
        entry = result[i] #type:dict
        corpus_id = entry["corpus_id"]
        score = entry["score"]
        print("----------------------------------")
        print("idx:", corpus_id, " score:", score)
        print(embdf["text"][corpus_id])

# 这个新闻是有关环保的，搜出来的结果确实都是环保相关的
search("Over the past decade, China has developed the world's largest clean coal-burning power generation base and built up the world's highest installed wind and photovoltaic power capacity. It has produced the most new energy vehicles globally for eight consecutive years.")
```

参考资料：

```
https://huggingface.co/sentence-transformers
https://zhuanlan.zhihu.com/p/457876366
```



