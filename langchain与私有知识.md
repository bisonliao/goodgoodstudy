### 问题提出

我有一组内部文档，包含大量的内部知识，如何利用chatGPT这样的大语言模型进行知识检索。

用embedding和KNN是一个方案，这里还有另外一个方案：

### 解决方案

把私有文档分割为chunk，分别embedding后存储到向量数据库。

输入一个查询，embedding后与文档的向量数据库做相似性比较，将相似性较高的chunk提取用作prompt优化，查询大语言模型。

引用网友的一个图：

![](img/llm/langchain.png)

### 参考资料

```shell
https://github.com/imClumsyPanda/langchain-ChatGLM
https://zhuanlan.zhihu.com/p/613155165
https://github.com/jerryjliu/llama_index
https://github.com/facebookresearch/faiss
```

### 亲测结果

#### llama_index

```shell
把白鹿原第一章内容喂给AI，然后
问：白嘉轩的父亲叫什么？
答：白嘉轩的父亲叫白秉德，他是白鹿原上的名医冷先生的父亲。当冷先生擦拭刀具时，秉德老汉的腿脚手臂松弛下来，口里开始淌出一股乌黑的粘液，令嘉轩感到恶心。秉德老汉在冷先生离开后不久去世，他的死亡是嘉轩头一回经见人的死亡过程

----AI明显混淆了，可能某个地方的“他”没有分清楚是冷先生还是白嘉轩，所以误认为也是冷先生的父亲。

```

#### chatGLM：

```shell
docker run --gpus all  -d --name chatglm -v /root/.cache:/root/.cache  -v /root/private:/private chatglm-cuda:latest  sh -c 'sleep 10000000'

root@23dec2396b03:/chatGLM# python3 ./cli_demo.py
Input your local knowledge file path 请输入本地知识文件路径：/private
加载文件: 100%|
文件加载完毕，正在生成向量库
Input your question 请输入问题：白嘉轩的职业是什么
白嘉轩的职业是白鹿原上一位富有的地主和娶过七房女人的多事男子。他通过购买鹿子霖家的土地获得了成功，并且娶到了多任妻子。他的人生充满了娶妻和生育，但是他也因为自己的贪心和冲动而面临了危险。

#把c++并发编程.pdf喂给AI，也就这个问题答得还可以，其他问题都一团糟

Input your question 请输入问题：使用c++ mutex时候，怎么确保及时释放mutex也就是及时unlock
在使用C++的 mutex 时，可以通过编写适当的代码来确保及时释放mutex,也就是及时unlock。

以下是一些实现的方法：

1. 使用std::lock_guard:

std::lock_guard可以使用类似于线程安全的锁来保护mutex。使用std::lock_guard可以在需要解锁mutex时进行解锁。例如：

​```
std::lock_guard<std::mutex> lock(mu);
​```

在代码中，使用std::lock_guard<std::mutex>类型的锁来保护mutex。当需要解锁mutex时，可以调用mu.unlock()方法来解锁。

2. 使用std::unique_lock:

std::unique_lock可以使用类似于线程安全的锁来保护mutex。使用std::unique_lock可以在需要解锁mutex时进行解锁。例如：

​```
std::unique_lock<std::mutex> lock(mu);
​```

在代码中，使用std::unique_lock类型的锁来保护mutex。在需要解锁mutex时，可以调用lock.Unlock()方法来解锁。

```

### fine tuning也是一个办法

如果私有知识可以整理成prompt/completion的形式（类似FAQ），也可以对模型进行fine tuning。调用openai的api接口很容易实现，但是我没有成功，提示欠费。

我在本地按照教程对chatGLM进行衣服广告词生成的fine tuning是成功的，fine tuning前后对比明显。详细可以见官方文档：

```shell
https://www.heywhale.com/mw/project/6436d82948f7da1fee2be59e
https://platform.openai.com/docs/api-reference/fine-tunes
```

对chatGLM进行fine tuning的时候，需要安装cuda的开发包：

```shell
apt install nvidia-cuda-dev
```

