### embedding与文本相关性搜索

#### 1、问题提出

一个技术服务公司的官网通常提供了他们产品的技术文档。如何提供一个好用的搜索？

#### 2、解决方案

用搜索引擎是一种方案，也有开源的可私有化部署的搜索引擎。结合最近很风光的openai公司的能力，我们可以有这样一个思路：

1. 用openai的开放的embedding接口和强大的文本理解能力，把文档转化为1536维大小的向量，并调用openai的接口生成文档的摘要
2. 利用redis/ElasticSearch等向量查询数据库，把文档的url和向量存储起来
3. 当用户输入一个查询语句，把它转化为1536维的向量，并到数据库里进行向量相似性搜索，找到相似的一批向量，把对应文档的url和摘要返回给用户

##### 2.1 数据预处理和embedding

```python
# -*- coding:utf-8 -*-
import pandas as pd
import csv
import os
import openai
import json
import tiktoken
import time

# openai embedding api处理的文本长度有限制，不超过8190个token
encoding = tiktoken.get_encoding("cl100k_base")
def cutStrByTokens(string: str) -> str:
    tokenList = encoding.encode(string)
    return encoding.decode(tokenList[:8190])

# 把要embedding的文本从文档中提取出来并去掉换行
text = list()
with open("E:\\新建文件夹\\飞书\\技术文档中文_20230418.csv", 'r', encoding='utf-8-sig') as f:
   r = csv.reader(f)
   for row in r:
       s = row[0] #type:str
       s = s.replace("\r\n", " ")
       s = s.replace("\n", " ")
       s = cutStrByTokens(s)
       if (len(s) < 3000):
           continue
       text.append(s)
print(len(text))

# embedding
openai.api_key = "sk-..."
emb=list()
for i in range(0, len(text)):
    s = text[i]#type:str
    resp = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text[i]
    )
    if  (i%7) == 5:
        time.sleep(1)
    print(i)
    j = json.loads(str(resp))
    if j["data"]:
        a = j["data"][0]["embedding"] #type:list
        #print(len(a))
        emb.append(a)
#存储到文件里
df = {"text":text, "emb":emb}
df = pd.DataFrame(df) #type:pd.DataFrame
df.to_pickle("e:\\save.pkl")
```



##### 2.2 数据入库

```python
import redis
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from redis.commands.search import Search
from redis.commands.search.query import Query
import numpy as np
import pandas as pd
import csv
import os
import openai
import json
import tiktoken
import time

#把文本原文和embedding后的向量入库
def load_vectors(client:redis.Redis, df, vector_field_name):
    p = client.pipeline(transaction=False)
    for index, row in df.iterrows():
        #hash key
        key='article:'+ str(index)
        #hash fields
        content = row['text']
        vector = np.array(row['emb']).astype(np.float32).tobytes()
        data_mapping ={'content':content, vector_field_name:vector}

        p.hset(key,mapping=data_mapping)
        if (index % 300) == 0:
            p.execute()
            p = client.pipeline(transaction=False)
    p.execute()

#Utility Functions to Create Indexes on Vector field
def create_flat_index (redis_conn,vector_field_name,number_of_vectors, vector_dimensions=1536, distance_metric='COSINE'):
    create_command =  ["FT.CREATE", "idx", "SCHEMA", "content","TEXT"]
    create_command += ["txtvector", "VECTOR", "FLAT", "8",
                        "TYPE", "FLOAT32",
                        "DIM", str(vector_dimensions),
                        "DISTANCE_METRIC", str(distance_metric),
                        "INITIAL_CAP", 300]
    redis_conn.execute_command(*create_command)

def create_hnsw_index (redis_conn,vector_field_name,number_of_vectors, vector_dimensions, distance_metric='COSINE',M=40,EF=200):

    create_command =  ["FT.CREATE", "idx", "SCHEMA", "content","TEXT"]
    create_command += ["txtvector", "VECTOR", "HNSW", "12",
                        "TYPE", "FLOAT32",
                        "DIM", str(vector_dimensions),
                        "DISTANCE_METRIC", str(distance_metric),
                        "INITIAL_CAP", 300,
                        "M", M,
                        "EF_CONSTRUCTION", EF]

    redis_conn.execute_command(*create_command)

    
df = pd.read_pickle('e:\\save.pkl')
host = '119.x.x.x'
port = 6379
redis_conn = redis.Redis(host = host, port = port)
print ('Connected to redis')

NUMBER_ARTICLES = 1800
VECTOR_FIELD_NAME = 'txtvector'
DISTANCE_METRIC = 'COSINE'
DIMENSIONS = 1536

redis_conn.flushall()
create_hnsw_index(redis_conn, VECTOR_FIELD_NAME, NUMBER_ARTICLES, DIMENSIONS, DISTANCE_METRIC)
load_vectors(redis_conn, df, VECTOR_FIELD_NAME)
print ('1800 News Articles loaded and indexed')

```



##### 2.3 数据查询

```python
# -*- coding:utf-8 -*-
import redis
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from redis.commands.search import Search
from redis.commands.search.query import Query
import json
import openai
import numpy as np
import time

openai.api_key = "sk-..."

user_query='服务端后台API，服务器异步回调'
resp = openai.Embedding.create(
    model="text-embedding-ada-002",
    input=user_query
)
j = json.loads(str(resp))
e = j["data"][0]["embedding"] #type:list
e = np.array(e).astype(np.float32) #type:np.array

print(len(e))


q = Query(f'*=>[KNN $K @txtvector $BLOB]').return_fields('content').dialect(2)

#connect to redis
host = '119.x.x.x'
port = 6379
redis_conn = redis.Redis(host = host, port = port)
print ('Connected to redis')

#parameters to be passed into search
params_dict = {"K": 4, "BLOB": e.tobytes()}
docs = redis_conn.ft().search(q,params_dict).docs
print("-----------------------------------")
for doc in docs:
    print ("********DOCUMENT: " + str(doc.id) + ' ********')
    print(doc.content)

print("\n\n")

user_query='开始混流，停止混流'
resp = openai.Embedding.create(
    model="text-embedding-ada-002",
    input=user_query
)
j = json.loads(str(resp))
e = j["data"][0]["embedding"] #type:list
e = np.array(e).astype(np.float32)
params_dict = {"K": 4, "BLOB": e.tobytes()}
docs = redis_conn.ft().search(q,params_dict).docs
print("-----------------------------------")
for doc in docs:
    print ("********DOCUMENT: " + str(doc.id) + ' ********')
    print(doc.content)


```

效果展示，可以看到有不错的相关性：

```shell
search   服务端后台API
1536
Connected to redis
-----------------------------------
********DOCUMENT: article:847 ********
# 调用方式 ---  ## 使用说明  ZEGO 服务端 API 支持 HTTPS 网络请求协议，允许 GET 或 POST 方法。您可以通过以下方式调用服务端 API：  - 根据 API 文档编写代码，访问相应 API。 - 参考 [使用 Postman 调测指南\|_blank](!Server_APIs_v2-Postman/Postman)，使用 ZEGO 提供的 Postman AP
********DOCUMENT: article:1006 ********
# 调测指南  - - - 在本文中我们为您介绍如何使用 Postman 调测服务端 API。  Postman 是一款 API 调测工具，可在让开发者在图形化界面中方便、直观地调测服务端 API。  为便于开发者调测云端录制的服务端 API，我们提供了对应的 Postman Collection，预先定义好了每个接口的请求参数，开发者导入后仅需修改参数取值即可调测。  ## 1 前提条件  -
********DOCUMENT: article:1078 ********
# 调测指南  - - -  在本文中我们为您介绍如何使用 Postman 调测服务端 API。  Postman 是一款 API 调测工具，可在让开发者在图形化界面中方便、直观地调测服务端 API。  为便于开发者调测超级白板的服务端 API，我们提供了对应的 Postman Collection，预先定义好了每个接口的请求参数，开发者导入后仅需修改参数取值即可调测。  ## 1 前提条件  -
********DOCUMENT: article:1204 ********
# 使用 Postman 调试  - - -  本文中将介绍如何使用 Postman 调试服务端 API。  Postman 是一款 API 调试工具，可在让开发者在图形化界面中方便、直观地调试服务端 API。  为便于开发者调试云通讯产品的服务端 API，ZEGO 提供了对应的 Postman Collection，预先定义好了每个接口的请求参数，开发者导入后仅需修改参数取值即可调试。  ##


search   开始混流，停止混流
-----------------------------------
********DOCUMENT: article:45 ********
# 混流 - - -  ## 1 功能简介 混流是把多路音视频流从云端混合成单流的技术。  ### 1.1 混流优点 1. 降低了开发实现上的复杂性。比如当有 N 个主播进行连麦，如果采用混流，观众端不必同时拉 N 路视频流，开发实现上省去了拉 N 路流并布局的步骤。 2. 降低了对设备的性能要求，减少设备的性能开销和网络带宽的负担。比如当连麦方过多时，观众端需要拉 N 路视频流，需要设备硬件上能
********DOCUMENT: article:82 ********
# 停止混流接口 - - -  > 请注意：该功能需要联系 ZEGO 技术支持开通。  ## 1 接口调用说明 > http请求方式: POST/FORM,需使用https <br>正式环境地址 <br>[https://webapi.zego.im/cgi/stop-mix?access_token=ACCESS_TOKEN | _blank](https://webapi.zego.im/cg
********DOCUMENT: article:202 ********
  # 停止混流接口 - - -  > 请注意：该功能需要联系 ZEGO 技术支持开通。  ## 1 接口调用说明 > http请求方式: POST/FORM,需使用https <br>正式环境地址 <br>[https://webapi.zego.im/cgi/stop-mix?access_token=ACCESS_TOKEN | _blank](https://webapi.zego.im/
********DOCUMENT: article:526 ********
# 多路混流 - - -  ## 1 功能简介 混流是把多路音视频流从云端混合成单流的技术。  ### 1.1 混流优点 1. 降低了开发实现上的复杂性。比如当有 N 个主播进行连麦，如果采用混流，观众端不必同时拉 N 路视频流，开发实现上省去了拉 N 路流并布局的步骤。 2. 降低了对设备的性能要求，减少设备的性能开销和网络带宽的负担。比如当连麦方过多时，观众端需要拉 N 路视频流，需要设备硬件

```

如果数据量不大，可以直接使用sentence-transformer库里的函数：

```python
util.semantic_search()
```

详细见：

```
https://www.sbert.net/examples/applications/semantic-search/README.html#util-semantic-search
```

参考资料：

```shell
https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
https://platform.openai.com/docs/api-reference/embeddings/create?lang=python
https://help.openai.com/en/articles/7437458-embeddings
https://github.com/RediSearch/RediSearch
https://hub.docker.com/r/redis/redis-stack
https://github.com/RedisAI/financial-news/blob/main/GettingStarted.ipynb
https://zhuanlan.zhihu.com/p/80737146
https://github.com/CLUEbenchmark/CLUE #这里可以下载到很多中文语料库
https://huggingface.co/sentence-transformers #这里有很多英文语料库
```



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
cc = torch.from_numpy(cc)
topk = torch.topk(cc, k= 4)
print(topk)
```

参考文档：

```
https://developer.nvidia.com/cuda-python
https://docs.cupy.dev/en/stable/install.html
https://docs.cupy.dev/en/stable/user_guide/kernel.html
```

#### 3、使用pretrained模型本地化部署来实现embedding也是可以的

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

    model = SentenceTransformer('paraphrase-MiniLM-L6-v2') #如果有GPU且torch的版本是带cuda的，会自动利用GPU
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

    #也可以这样写，支持一次encoding 多个sentence
def embedNews():
    # dataset from https://huggingface.co/datasets/argilla/news-summary
    dataset = pq.ParquetDataset("E:\\Temp\\download\\test-00000-of-00001-6227bd8eb10a9b50.parquet")
    df = dataset.read_pandas().to_pandas() #type:pd.DataFrame
    print(df.shape)
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    input = df.get("text").values.tolist()
    embList = model.encode(input) #type:np.ndarray
    embList = embList.tolist()
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
https://zhuanlan.zhihu.com/p/152522906
```



